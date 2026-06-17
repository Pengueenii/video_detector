import os
import argparse
import datetime
import cv2
import click
from urllib.parse import urlparse
import yt_dlp
import yaml
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

YDLP_OPTS = {"format": "best", "quiet": True}

script_location = Path(__file__).parent
config_folder = script_location / "configs"


@dataclass
class Reason:
    name: str
    template: cv2.typing.MatLike
    interval: int = 60
    threshold: float = 0.2
    x_min: Optional[int] = None
    x_max: Optional[int] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    prev_frame: Optional[cv2.typing.MatLike] = None
    match_percentage: float = 0
    found: bool = False


@dataclass
class GameConfig:
    name: str
    reasons: list = field(default_factory=list)


def get_configs() -> list[str]:
    ret = []
    for f in config_folder.iterdir():
        if f.is_dir():
            ret.append(f.name)

    return ret


def load_game(name: str):
    game_dir = config_folder / name
    config_file = game_dir / "config.yaml"
    with open(config_file) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    gameconfig = GameConfig(data["game"])

    for item in data["reasons"]:
        template = cv2.imread(game_dir / item["template"])
        if template is None:
            raise ValueError("Failed to read reference image")
        name = item["name"]
        interval = item["interval"]
        threshold = item["threshold"]

        reason = Reason(name, template, interval, threshold)

        if "zone" in item:
            reason.x_min = item["zone"]["x-min"]
            reason.x_max = item["zone"]["x-max"]
            reason.y_min = item["zone"]["y-min"]
            reason.y_max = item["zone"]["y-max"]

        gameconfig.reasons.append(reason)

    return gameconfig


def time_to_ms(time: str) -> int:
    dt = datetime.datetime.strptime(time, "%H:%M:%S")

    return ((dt.hour * 3600) + (dt.minute * 60) + dt.second) * 1000


def fetch_vod(url: str) -> str:
    with yt_dlp.YoutubeDL(YDLP_OPTS) as vod:
        print("Fetching VOD metadata")
        info = vod.extract_info(url, download=False)
        stream_url = info.get("url")
        if not stream_url:
            raise ValueError("No video URL found")
        return stream_url


def initialize_video(url: str) -> str:
    res = urlparse(url)
    if not res.hostname or "twitch.tv" not in res.hostname:
        raise ValueError("Url is not a valid twitch url")

    if not res.path:
        raise ValueError("No path provded in url")

    if "videos" in res.path or "clip" in res.path:
        print("Processing VOD")
        return fetch_vod(url)

    return ""


def process_frame(
    capture: cv2.VideoCapture, frame_number: int, output: Path, reasons: list[Reason]
):
    msec = capture.get(cv2.CAP_PROP_POS_MSEC)
    vid_time = str(datetime.timedelta(milliseconds=msec)).split(".")[0]

    print(
        f"\rTime in video: {vid_time}",
        end="",
        flush=True,
    )

    for reason in reasons:
        if frame_number % reason.interval == 0:
            _, frame = capture.retrieve()

            roi = frame
            if reason.x_min and reason.x_max and reason.y_min and reason.y_max:
                roi = frame[reason.y_min : reason.y_max, reason.x_min : reason.x_max]

            res = cv2.matchTemplate(roi, reason.template, cv2.TM_CCOEFF_NORMED)
            max_val = cv2.minMaxLoc(res)[1]
            is_found = max_val >= reason.threshold

            if is_found and max_val > reason.match_percentage:
                reason.prev_frame = frame

            if reason.found and not is_found:
                print(" Found", reason.name)
                cv2.imwrite(
                    filename=f"{output}/{reason.name}_{vid_time.replace(':', '_')}.jpg",
                    img=reason.prev_frame,
                )

                reason.match_percentage = 0
                reason.prev_frame = None

            reason.found = is_found


def process_video(video: str, config: GameConfig, time_ms: int, output: Path):
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        print("Could not open stream link")
        return

    capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)

    output.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    success = True
    while success:
        success = capture.grab()
        process_frame(capture, frame_count, output, config.reasons)
        frame_count += 1


@click.command
@click.option("-t", "--time", help="time to start at (HH:MM:SS)", default="00:00:00")
@click.option(
    "-o",
    "--output",
    help="Folder to output files",
    default="frames",
    type=click.Path(path_type=Path),
)
@click.option(
    "-g",
    "--game",
    type=click.Choice(get_configs()),
    help="Choosing a game helps remove false positives",
)
@click.argument(
    "VIDEO_URL",
)
def main(time: str, video_url: str, game: str, output: Path):
    """
    Parses through a video finding occurences of reference images, this program assumes the video is 1920x1080
    """

    start = datetime.datetime.now()
    gameconfig = load_game(game)
    vid = initialize_video(video_url)
    time_ms = time_to_ms(time)

    try:
        process_video(vid, gameconfig, time_ms, output)
    except KeyboardInterrupt:
        pass

    print("\n", "=" * 50, sep="")
    elapsed = datetime.datetime.now() - start
    print(f"Time elapsed: {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
