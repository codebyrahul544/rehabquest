"""Full app, headless: calibration -> menu -> 5 tutorials -> 5 games -> results, in demo mode."""
import main


def test_full_journey_selftest():
    assert main.main(["--selftest"]) == 0
