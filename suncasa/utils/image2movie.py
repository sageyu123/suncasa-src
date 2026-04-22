import argparse
import subprocess
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_movie_from_images(
        input_path,
        output_name='output_movie',
        image_format='png',
        frame_rate=10,
        resolution=None,
        quality=15,
        start_frame=0,
        crop_region=None,
        timestamps=None,
        image_dpi=200,
        preserve_temp=False,
        use_existing_temp=False,
        allow_rotation=True,
        force_overwrite=False
):
    """
    Create a movie from a series of images.

    :param str input_path: Path to image files or list of image paths
    :param str output_name: Name of the output movie file (without extension)
    :param str image_format: Format of the input image files
    :param int frame_rate: Frames per second in the output movie
    :param str resolution: Output movie resolution (format: 'widthxheight')
    :param int quality: Video quality (0-51, lower is better)
    :param int start_frame: Starting number for image sequence
    :param list crop_region: Crop images: [left, right, top, bottom]
    :param list timestamps: List of timestamps for each frame
    :param int image_dpi: DPI for image processing
    :param bool preserve_temp: Keep temporary files after processing
    :param bool use_existing_temp: Use existing images in the temp folder
    :param bool allow_rotation: Allow automatic image rotation
    :param bool force_overwrite: Overwrite existing output file
    :return: None
    """
    if isinstance(input_path, list):
        image_files = input_path
        base_dir = os.path.dirname(input_path[0])
    else:
        image_files = sorted(glob.glob(f"{input_path}*.{image_format}"))
        base_dir = os.path.dirname(input_path)

    if not image_files:
        print('No images found!')
        return

    temp_dir = os.path.join(base_dir, 'temp_images') + os.path.sep
    os.makedirs(temp_dir, exist_ok=True)

    if not use_existing_temp:
        process_images(image_files, temp_dir, image_format, crop_region, timestamps, resolution, image_dpi)

    ffmpeg_options = {
        'r': frame_rate,
        's': resolution,
        'start_number': start_frame,
        'crf': quality
    }
    if resolution is None:
        ffmpeg_options.pop('s')

    overwrite_flag = '-y' if force_overwrite else '-n'
    rotation_flag = '' if allow_rotation else '-noautorotate'

    ffmpeg_args = ' '.join([f'-{k} {v}' for k, v in ffmpeg_options.items()])

    output_path = f"{os.path.join(base_dir, output_name)}"
    if not output_path.lower().endswith('.mp4'):
        output_path += '.mp4'

    if not force_overwrite and os.path.exists(output_path):
        print(f"Error: Output file '{output_path}' already exists. Use -y or --force-overwrite to overwrite.")
        return

    overwrite_flag = '-y' if force_overwrite else '-n'
    rotation_flag = '' if allow_rotation else '-noautorotate'

    ffmpeg_command = (
        f'ffmpeg {rotation_flag} {overwrite_flag} -r {frame_rate} -f image2 '
        f'-i {temp_dir}%04d.{image_format} '
        f'-vcodec libx264 -pix_fmt yuv420p {ffmpeg_args} '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        f'{output_path}'
    )

    try:
        subprocess.check_output(['bash', '-c', ffmpeg_command], stderr=subprocess.STDOUT)
        print(f"Movie created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating movie: {e.output.decode()}")

    print(f"FFmpeg command: {ffmpeg_command}")

    if not preserve_temp:
        os.system(f'rm -rf {temp_dir}')


def process_images(image_files, temp_dir, image_format, crop_region, timestamps, resolution, image_dpi):
    if crop_region or timestamps:
        plt.ioff()
        fig_size = [float(dim) / image_dpi for dim in resolution.split('x')] if resolution else None
        fig, ax = plt.subplots(figsize=fig_size)
        for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
            data = plt.imread(img_path)
            if crop_region:
                left, right, top, bottom = crop_region
                data = data[top:bottom + 1, left:right + 1, :]
            if idx == 0:
                im = ax.imshow(data)
                ax.set_axis_off()
            else:
                im.set_array(data)
            if timestamps:
                ax.set_title(timestamps[idx])
            if idx == 0:
                fig.tight_layout()
            fig.savefig(f'{temp_dir}{idx:04d}.{image_format}', dpi=image_dpi, format=image_format)
        plt.close(fig)
    else:
        for idx, img_path in enumerate(tqdm(image_files, desc="Copying images")):
            os.system(f'cp {img_path} {temp_dir}{idx:04d}.{image_format}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a movie from a series of images.")
    parser.add_argument("input_path", help="Path to image files or list of image paths")
    parser.add_argument("-o", "--output-name", default="output_movie",
                        help="Name of the output movie file (without extension)")
    parser.add_argument("-f", "--image-format", default="png", help="Format of the input image files")
    parser.add_argument("-r", "--frame-rate", type=int, default=10, help="Frames per second in the output movie")
    parser.add_argument("-s", "--resolution", help="Output movie resolution (format: 'widthxheight')")
    parser.add_argument("-q", "--quality", type=int, default=15, help="Video quality (0-51, lower is better)")
    parser.add_argument("-n", "--start-frame", type=int, default=0, help="Starting number for image sequence")
    parser.add_argument("-c", "--crop-region", nargs=4, type=int, help="Crop images: left right top bottom")
    parser.add_argument("-t", "--timestamps", nargs="+", help="List of timestamps for each frame")
    parser.add_argument("-d", "--image-dpi", type=int, default=200, help="DPI for image processing")
    parser.add_argument("-p", "--preserve-temp", action="store_true", help="Keep temporary files after processing")
    parser.add_argument("-u", "--use-existing-temp", action="store_true", help="Use existing images in the temp folder")
    parser.add_argument("--no-rotation", dest="allow_rotation", action="store_false",
                        help="Disable automatic image rotation")
    parser.add_argument("-y", "--force-overwrite", action="store_true", help="Overwrite existing output file")

    args = parser.parse_args()

    create_movie_from_images(
        input_path=args.input_path,
        output_name=args.output_name,
        image_format=args.image_format,
        frame_rate=args.frame_rate,
        resolution=args.resolution,
        quality=args.quality,
        start_frame=args.start_frame,
        crop_region=args.crop_region,
        timestamps=args.timestamps,
        image_dpi=args.image_dpi,
        preserve_temp=args.preserve_temp,
        use_existing_temp=args.use_existing_temp,
        allow_rotation=args.allow_rotation,
        force_overwrite=args.force_overwrite
    )