#!/usr/bin/env python3
"""Read selected SwanLab metrics and check profile redaction without exporting profiles."""
import argparse
import json
import os
from pathlib import Path
import re

METRIC_KEYS = (
    'actor/lr', 'actor/kl_coef', 'actor/grad_norm',
    'training/zero_adv_group_fraction', 'training/n_model_output_tokens',
    'training/n_optimizer_action_tokens',
)
SECRET_FIELD = re.compile(r'api.?key|credential|password|secret|auth.?token|access.?token|service.?key|^AGL_KEY$', re.I)


def inspect_redaction(sections):
    checked, unsafe, missing = [], [], []
    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                current = path + '.' + str(key)
                if SECRET_FIELD.search(str(key)):
                    checked.append(current)
                    masked = isinstance(child, str) and (
                        child.strip().lower() in {'redacted', '<redacted>', '[redacted]', '***redacted***', '<masked>', '[masked]'}
                        or bool(child) and set(child) == {'*'}
                    )
                    if child not in (None, '', False) and not masked:
                        unsafe.append(current)
                else:
                    walk(child, current)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f'{path}[{index}]')
    for section, value in sections.items():
        if value is None:
            missing.append(section)
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                unsafe.append(section + '.unparsed_profile')
                continue
        walk(value, section)
    return {'checked_sections': list(sections), 'sensitive_field_count': len(checked),
            'unsafe_field_names': unsafe, 'missing_sections': missing, 'profile_values_exported': False}


def read_run(run, tail=4):
    if tail < 1:
        raise ValueError('tail must be positive')
    metrics = run.metrics(keys=list(METRIC_KEYS))
    sections = {name: getattr(run.profile, name, None) for name in ('config', 'metadata')}
    return {'run_id': run.id, 'cloud_state': str(run.state),
            'metrics': json.loads(metrics.tail(tail).to_json(orient='index')),
            'redaction': inspect_redaction(sections),
            'scope': 'Cloud readback only; verify real processes and checkpoints separately. No model-quality conclusion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='SwanLab workspace/project/run-id path')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; existing files are preserved')
    parser.add_argument('--tail', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new report path')
    import swanlab
    report = read_run(swanlab.Api(api_key=os.environ.get('SWANLAB_API_KEY')).run(args.run), args.tail)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write('\n')
    print(json.dumps({'run_id': report['run_id'], 'cloud_state': report['cloud_state'],
                      'metric_rows': len(report['metrics']), 'redaction': report['redaction']}))
    return 2 if report['redaction']['unsafe_field_names'] or report['redaction']['missing_sections'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
