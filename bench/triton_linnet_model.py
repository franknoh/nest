"""Triton Inference Server's Python backend serving a Linnet decoder through
`linnet.serve`: requests join the engine's rows as they arrive (continuous
batching), in a thread of its own, and each is answered as it runs -- its
first token as soon as it exists, all of them when it finishes. Decoupled,
so a request's responses are sent from that thread, not returned by
`execute`.

Written into the model repository by `bench.methods.serving`, which fills
in `CARD` and `GENERICS`. Inputs: `prompt` (INT32 token ids) and
`max_tokens` (INT32, one value); output: `tokens` (INT32).
"""

import queue
import threading

import numpy as np
import triton_python_backend_utils as pb_utils

CARD = "__CARD__"
GENERICS: dict = {}  # __GENERICS__


class TritonPythonModel:
    def initialize(self, args):
        from linnet import nest
        from linnet.serve import Engine

        model = nest.load(
            CARD,
            backend="torch",
            device="cuda",
            numerics="fast",
            compile=True,
            generics=GENERICS,
            cast_dtype=True,
        )
        self.engine = Engine(model)
        # Every prompt length the engine compiles, so no request waits on one.
        self.engine.warmup(self.engine.buckets)
        self.incoming = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def execute(self, requests):
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy().reshape(-1)
            budget = pb_utils.get_input_tensor_by_name(request, "max_tokens").as_numpy()
            self.incoming.put(
                (
                    request.get_response_sender(),
                    [int(t) for t in prompt],
                    int(budget.reshape(-1)[0]),
                )
            )
        return None

    def admit(self, item, open_requests):
        from linnet.serve import Request

        sender, prompt, budget = item
        completion = self.engine.submit(Request(prompt=prompt, max_new_tokens=budget))
        open_requests.append([sender, completion, False])

    def loop(self):
        open_requests = []  # [sender, completion, first token sent]
        while self.running:
            # Idle: wait for a request. Busy: take whatever has arrived.
            try:
                item = (
                    self.incoming.get(timeout=0.05)
                    if not self.engine.busy
                    else self.incoming.get_nowait()
                )
                while True:
                    self.admit(item, open_requests)
                    item = self.incoming.get_nowait()
            except queue.Empty:
                pass
            if not self.engine.busy:
                continue
            finished = {id(c) for c in self.engine.step()}
            still_open = []
            for entry in open_requests:
                sender, completion, first_sent = entry
                if id(completion) in finished:
                    tokens = pb_utils.Tensor(
                        "tokens", np.asarray(completion.tokens, dtype=np.int32)
                    )
                    sender.send(
                        pb_utils.InferenceResponse(output_tensors=[tokens]),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )
                    continue
                if not first_sent and completion.tokens:
                    first = pb_utils.Tensor(
                        "tokens", np.asarray(completion.tokens[:1], dtype=np.int32)
                    )
                    sender.send(pb_utils.InferenceResponse(output_tensors=[first]))
                    entry[2] = True
                still_open.append(entry)
            open_requests = still_open

    def finalize(self):
        self.running = False
