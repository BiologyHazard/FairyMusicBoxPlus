import itertools
from pathlib import Path

from main import main, subparsers
from music_box_designer.recognize import Note, export_midi, recognize_multi_image, recognize_pdf
from music_box_designer.draft import find_available_filename


def recognize_func(args):
    source_path = Path(args.source)
    if source_path.suffix == '.pdf':
        result: list[Note] = recognize_pdf(source_path)
    else:
        image_paths: list[Path] = []
        for i in itertools.count(0):
            image_path = Path(args.source.format(i))
            if image_path.is_file():
                image_paths.append(image_path)
            elif i > 1:  # 允许图片从1开始
                break
        if not image_paths:
            raise FileNotFoundError(f'No such file or directory: {args.source}')
        result = recognize_multi_image(image_paths)

    if args.destination is not None:
        destination_path = Path(args.destination)
    else:
        destination_path = Path(args.source).with_suffix('.mid')

    export_midi(result, args.quantization).save(find_available_filename(destination_path, args.overwrite))


recognize_parser = subparsers.add_parser('recognize', help='Recognize notes from draft.')
recognize_parser.set_defaults(func=recognize_func)
recognize_parser.add_argument(
    'source',
    type=str,
)
recognize_parser.add_argument(
    'destination',
    type=str,
    nargs='?',
    default=None,
)
# recognize_parser.add_argument('-t', '--transposition', type=int, default=0)
recognize_parser.add_argument('-o', '--overwrite', action='store_true')
recognize_parser.add_argument('-q', '--quantization', type=float, default=1/4)

if __name__ == '__main__':
    main()
