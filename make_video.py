import argparse
import io
import os
import subprocess
import zipfile

import requests
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ImageEvent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--key", default="generated")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--n_frames", type=int)
    parser.add_argument("--ffmpeg_args", default="-c:v libx264 -pix_fmt yuv420p -preset veryslow")
    args = parser.parse_args()

    assert not os.path.exists(args.output_path), f"{args.output_path} exists"

    event_acc = EventAccumulator(args.tb_path, size_guidance=dict(images=0))
    event_acc.Reload()
    print(event_acc.Tags())

    img_events: list[ImageEvent] = event_acc.Images(args.key)
    width = img_events[0].width
    height = img_events[0].height

    font_path = "fonts/Roboto_Mono/RobotoMono-VariableFont_wght.ttf"
    font_dir = os.path.dirname(font_path)
    if not os.path.exists(font_dir):
        resp = requests.get("https://fonts.google.com/download?family=Roboto%20Mono")
        resp.raise_for_status()
        zip = zipfile.ZipFile(io.BytesIO(resp.content))
        zip.extractall(font_dir)
    font = ImageFont.truetype(font_path, size=20)

    _, text_height = font.getsize("Step")
    canvas_height = height + text_height

    cmd = (
        "ffmpeg"
        + f" -f rawvideo -pixel_format rgb24 -video_size {width}x{canvas_height} -framerate {args.fps}"
        + f" -i pipe: {args.ffmpeg_args} {args.output_path}"
    )
    proc = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE)

    if args.n_frames is not None:
        if len(img_events) < args.n_frames:
            print(f"There are only {len(img_events)} frames. {args.n_frames=} will have no effect")
        else:
            img_events = img_events[: args.n_frames]

    for img_evt in img_events:
        canvas = Image.new("RGB", (width, height + text_height), (255, 255, 255))
        img = Image.open(io.BytesIO(img_evt.encoded_image_string))
        canvas.paste(img, (0, 0))

        step = img_evt.step
        draw = ImageDraw.Draw(canvas)
        draw.text((0, canvas_height), f"Step {step:06d}", font=font, fill=(0, 0, 0), anchor="lb")

        if proc.poll() is not None:
            print("ffmpeg process ends early. exiting...")
            break
        proc.stdin.write(canvas.tobytes())

    proc.stdin.close()


if __name__ == "__main__":
    main()
