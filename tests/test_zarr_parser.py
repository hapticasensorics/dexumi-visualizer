from pathlib import Path

from dexumi_visualizer.zarr_parser import discover_episodes, episode_summary


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "data" / "sample_data"


def test_discover_episodes_dataset() -> None:
    dataset = SAMPLE_DATA / "software_go_through"
    episodes = discover_episodes(dataset)
    names = {episode.name for episode in episodes}
    assert {"episode_0", "episode_1"}.issubset(names)


def test_discover_episodes_root() -> None:
    episodes = discover_episodes(SAMPLE_DATA)
    assert any(
        episode.name == "episode_0" and "software_go_through" in episode.parts
        for episode in episodes
    )


def test_episode_summary() -> None:
    episode = SAMPLE_DATA / "software_go_through" / "episode_0"
    summary = episode_summary(episode)
    sensor_names = {sensor.name for sensor in summary.sensors}
    assert "numeric_0" in sensor_names
    assert "camera_0" in sensor_names
    assert summary.total_frames is None or summary.total_frames > 0
    if summary.duration_s is not None:
        assert summary.duration_s > 0


def test_episode_summary_video_sensors() -> None:
    episode = SAMPLE_DATA / "software_go_through_replay" / "episode_0"
    summary = episode_summary(episode)
    sensor_names = {sensor.name for sensor in summary.sensors}
    assert "dex_camera_0" in sensor_names
    assert "exo_camera_0" in sensor_names
