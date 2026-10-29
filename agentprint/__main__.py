"""Only approved runtime options are accepted at the public command boundary."""
import argparse
import json
import sys

from .config import ConfigError, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m agentprint')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'train', 'evaluate'):
        child = commands.add_parser(command)
        child.add_argument('--config', required=True)
        for option in ('data-root', 'cache-root', 'output-root'):
            child.add_argument('--' + option)
        if command in ('train', 'evaluate'):
            child.add_argument('--device', default='cpu')
        if command == 'train':
            child.add_argument('--resume')
        if command == 'evaluate':
            child.add_argument('--checkpoint', required=True)
            child.add_argument('--split', choices=('val', 'test'), default='val')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, {key: getattr(args, key) for key in
                                           ('data_root', 'cache_root', 'output_root')})
        if args.command == 'prepare':
            from .preparation import prepare
            result = prepare(config)
            print(json.dumps({'complete': result['complete'], 'datasets': result['datasets']}))
        elif args.command == 'train':
            from .training import train
            print(json.dumps({'run_dir': str(train(config, args.device, args.resume))}))
        else:
            from .training import evaluate
            print(json.dumps(evaluate(config, args.device, args.checkpoint, args.split)))
        return 0
    except (ConfigError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f'agentprint: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
