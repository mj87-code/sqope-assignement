"""
Interactive query client — ask questions against the running API.

Run it as an interactive container (connects to the same compose network):

    docker run -it --rm \
      --network sqope-ai-home-assignment-task_default \
      -e API_URL=http://api:8000 \
      sqope-ai-home-assignment-task-indexer \
      python ask.py

Uses only the Python standard library, so any of the project images can run it.
Set API_URL to point at the API (default http://localhost:8000).
Set SHOW_TRACE=1 to also print each answer's internal pipeline steps.
"""
import itertools
import json
import os
import sys
import threading
import time
import urllib.request

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
SHOW_TRACE = os.environ.get("SHOW_TRACE", "").lower() in {"1", "true", "yes"}


class Spinner:
    """Minimal stdlib spinner shown while a blocking call runs."""

    def __init__(self, message: str = "thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin)

    def __enter__(self):
        # Only animate on a real terminal; piped/redirected output stays clean.
        if sys.stdout.isatty():
            self._thread.start()
        return self

    def _spin(self):
        for frame in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{self.message} {frame}")
            sys.stdout.flush()
            time.sleep(0.1)

    def __exit__(self, *exc):
        if not self._thread.is_alive():
            return
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()


def ask(question: str) -> dict:
    data = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        f"{API_URL}/query?verbose=true",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def _show(result: dict) -> None:
    if SHOW_TRACE:
        trace = result.get("trace") or []
        if trace:
            print("\nPipeline steps:")
            for line in trace:
                print(f"  {line}")
        print(
            f"\n[{result['query_type']}] "
            f"eval_passed={result['eval_passed']} "
            f"confidence={result['confidence']:.2f}"
        )

    basis = result.get("answer_basis")
    if result.get("answer"):
        print(f"\n{result['answer']}\n")
    elif basis == "out_of_scope":
        print(f"\n(out of scope) {result.get('rejection_reason')}\n")
    elif basis == "needs_clarification":
        print(f"\n(clarification needed) {result.get('rejection_reason')}")
        print(
            "This system has no conversation memory — type your NEXT question as a "
            "full, self-contained question (including the missing detail above), not "
            "a short reply.\n"
        )
    elif SHOW_TRACE:
        print(f"\n(no answer) {result.get('rejection_reason')}\n")
    else:
        print("\n(no answer) No sufficiently relevant information was found for this "
              "question. Try rephrasing it or asking about something more specific.\n")

    res = result.get("result")
    if res:
        computed = res.get("computed")
        if computed:
            print(
                f"Verified ({res['kind']}): "
                f"{computed['operation']}({computed.get('column')}) = {computed['value']}"
            )
        if res.get("rows") is not None:
            print(f"Verified rows ({res['kind']}): {res['rows']}")
        if res.get("sql"):
            print(f"SQL: {res['sql']}")

    for s in result.get("sources", []):
        page = f", p.{s['page_number']}" if s.get("page_number") else ""
        print(f"  - {s['doc_filename']}{page}: {s['content_snippet'][:100]}")


def main() -> None:
    print(f"Connected to {API_URL}. Ask a question (Ctrl-D or 'quit' to exit).")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break
        try:
            with Spinner("thinking"):
                result = ask(q)
            _show(result)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
