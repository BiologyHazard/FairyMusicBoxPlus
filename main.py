import argparse

from music_box_designer import convert, generate_draft, get_note_count_and_length, logger
from music_box_designer.log import set_level


def convert_func(args) -> None:
    return convert(args.source, args.destination, transposition=args.transposition, overwrite=args.overwrite)


def draft_func(args) -> None:
    return generate_draft(
        source_path=args.file_path,
        settings_path=args.settings_path,
        pdf=args.pdf,
        note_count=args.note_count,
        transposition=args.transposition,
        remove_blank=not args.keep_blank,
        skip_near_notes=not args.keep_near_notes,
        bpm=args.bpm,
        title=args.title,
        subtitle=args.subtitle,
        music_info=args.music_info,
        tempo_text=args.tempo_text,
        scale=args.scale,
        overwrite=args.overwrite,
    )


def count_func(args) -> None:
    note_count, length_mm = get_note_count_and_length(
        file_path=args.file_path,
        transposition=args.transposition,
        remove_blank=not args.keep_blank,
        skip_near_notes=not args.keep_near_notes,
        bpm=args.bpm,
        scale=args.scale,
    )
    print(f'Notes: {note_count}')
    print(f'Length: {length_mm / 1000:.2f}m')


parser = argparse.ArgumentParser(description='Music Box Designer')
subparsers = parser.add_subparsers(title='commands', required=True)
parser.add_argument('--log-level',
                    type=lambda s: str(s).upper(),
                    default='DEBUG',
                    choices=['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'])
convert_parser = subparsers.add_parser(
    'convert',
    help='Convert .emid / .fmp / .mid file to another format.',
    description='Convert .emid / .fmp / .mid file to another format.',
    epilog=f'''examples:
  {parser.prog} convert examples/example.emid examples/example.mid
  {parser.prog} convert examples/example.emid .mid    # equivalent to the previous command
  {parser.prog} convert examples/*.mid examples/*.fmp
  {parser.prog} convert examples/*.mid .fmp    # equivalent to the previous command''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
convert_parser.set_defaults(func=convert_func)
convert_parser.add_argument(
    'source',
    type=str,
    help="Specify the source file path, or just the extension (including the leading '.', e.g. 'directory/*.fmp', "
         "which will convert all .fmp files in directory/).",
)
convert_parser.add_argument(
    'destination',
    type=str,
    help="If source specified a file, destination can either be a path or an extension. If source provided an "
         "extension, destination should be a different extension.",
)
convert_parser.add_argument('-t', '--transposition', type=int, default=0)
convert_parser.add_argument('-o', '--overwrite', action='store_true')

draft_parser = subparsers.add_parser('draft', help='Generate draft pics.')
draft_parser.set_defaults(func=draft_func)
draft_parser.add_argument('file_path', type=str)
draft_parser.add_argument('settings_path', type=str, nargs='?', default='draft_settings.yml')
draft_parser.add_argument('-p', '--pdf', action='store_true')
draft_parser.add_argument('-N', '--note-count', type=int)
draft_parser.add_argument('-t', '--transposition', type=int, default=0)
draft_parser.add_argument('-k', '--keep-blank', action='store_true')
draft_parser.add_argument('-n', '--keep-near-notes', action='store_true')
draft_parser.add_argument('-b', '--bpm', type=float, default=None)
draft_parser.add_argument('-T', '--title', type=str)
draft_parser.add_argument('-S', '--subtitle', type=str)
draft_parser.add_argument('-I', '--music-info', type=str)
draft_parser.add_argument('-B', '--tempo-text', type=str)
draft_parser.add_argument('-s', '--scale', type=float, default=1)
draft_parser.add_argument('-o', '--overwrite', action='store_true')

count_parser = subparsers.add_parser('count', help='Count notes and length.')
count_parser.set_defaults(func=count_func)
count_parser.add_argument('file_path', type=str)
count_parser.add_argument('-t', '--transposition', type=int, default=0)
count_parser.add_argument('-k', '--keep-blank', action='store_true')
count_parser.add_argument('-n', '--keep-near-notes', action='store_true')
count_parser.add_argument('-b', '--bpm', type=float, default=None)
count_parser.add_argument('-s', '--scale', type=float, default=1)


@logger.catch
def main() -> None:
    args = parser.parse_args()
    set_level(args.log_level)
    result = args.func(args)
    logger.success('Done!')
    return result


if __name__ == '__main__':
    main()
