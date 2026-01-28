from pathlib import Path

from typer.testing import CliRunner

from dexumi_visualizer.cli import app


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "data" / "sample_data"


def test_cli_list() -> None:
    runner = CliRunner()
    dataset = SAMPLE_DATA / "software_go_through"
    result = runner.invoke(app, ["list", str(dataset)])
    assert result.exit_code == 0
    assert "episode_0" in result.stdout


def test_cli_info() -> None:
    runner = CliRunner()
    episode = SAMPLE_DATA / "software_go_through" / "episode_0"
    result = runner.invoke(app, ["info", str(episode)])
    assert result.exit_code == 0
    assert "Episode: episode_0" in result.stdout
    assert "Sensors" in result.stdout
