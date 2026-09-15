from app.engine import IgnitionEngine
from app.evaluation import corpus_m2


def test_m2_rejects_activity_without_seed_dependencies(monkeypatch):
    original = IgnitionEngine.run
    def disconnected(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.run = result.run.model_copy(update={"ticks": tuple(event.model_copy(update={"parent_trace": None, "parent_traces": ()}) for event in result.run.ticks)})
        return result
    monkeypatch.setattr(IgnitionEngine, "run", disconnected)
    report = corpus_m2()
    assert report["passed"] == 0 and report["trace_pass_rate"] == 0
