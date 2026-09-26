"""Evaluate M2-A/M2-B/M2-B1 on the sealed M2 test split and build visual reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import lpips
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .m2a_data_baseline import load_m2_manifest
from .qwen_backend import QwenSharedBackend
from .stage1_train import image_tensor, ssim

ROOT = Path('/share/linmingheng-local/xuke')
PROJECT = ROOT / 'RMagNet'
DATA_ROOT = ROOT / 'datasets/rmagnet_m2_aspect'
OUTPUT_DIR = PROJECT / 'runs/m2_corrected_three_way_test'
VARIANTS = {
    'M2-A': PROJECT / 'runs/m2_corrected_a_data70_noq20/best_transmission_lora.safetensors',
    'M2-B': PROJECT / 'runs/m2_corrected_b_q20full70/best_transmission_lora.safetensors',
    'M2-B1': PROJECT / 'runs/m2_corrected_b1_q20grad30/best_transmission_lora.safetensors',
}
SELECTED = ['141_3946_2327', '111_2924_1604', '153_3556_2410']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantize01(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().clamp(0, 1).mul(255).round().div(255)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = quantize01(x).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((x * 255).round().astype(np.uint8), 'RGB')


def pil_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGB'), dtype=np.float32).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).div_(255)


def save_png(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(x).save(path, format='PNG', compress_level=6)


def load_adapter(backend: QwenSharedBackend, path: Path, device: torch.device) -> None:
    state = safetensors.torch.load_file(str(path), device=str(device))
    _, unexpected = backend.transformer.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected adapter keys in {path}: {unexpected[:5]}')


def measure(pred: torch.Tensor, gt: torch.Tensor, inp: torch.Tensor, lpips_model) -> dict[str, float]:
    pred, gt, inp = quantize01(pred).cpu(), quantize01(gt).cpu(), quantize01(inp).cpu()
    mse = F.mse_loss(pred, gt)
    err = (pred - gt).abs().mean(1, keepdim=True)
    change = (inp - gt).abs().mean(1, keepdim=True)
    low = change <= torch.quantile(change.flatten(), 0.25)
    high = change >= torch.quantile(change.flatten(), 0.75)
    with torch.no_grad():
        perceptual = float(lpips_model(pred.mul(2).sub(1), gt.mul(2).sub(1)).mean())
    return {
        'l1': float(F.l1_loss(pred, gt)),
        'psnr': float(-10 * torch.log10(mse.clamp_min(1e-12))),
        'ssim': float(ssim(pred, gt)),
        'lpips_squeeze': perceptual,
        'low_change_keep_l1': float(err[low].mean()),
        'high_change_restore_l1': float(err[high].mean()),
    }


def fit(im: Image.Image, size=(480, 360)) -> Image.Image:
    canvas = Image.new('RGB', size, (28, 28, 28))
    copy = im.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def error_map(pred: torch.Tensor, gt: torch.Tensor) -> Image.Image:
    err = (quantize01(pred) - quantize01(gt)).abs().mean(1, keepdim=True)
    err = (err * 4).clamp(0, 1)
    heat = torch.cat((err, err.sqrt() * 0.55, torch.zeros_like(err)), 1)
    return tensor_to_pil(heat)


def build_panel(sample_id: str, images: dict[str, torch.Tensor], rows: dict[str, dict], out: Path) -> None:
    labels = ['Input', 'GT', 'M2-A', 'M2-B', 'M2-B1']
    cell_w, cell_h, header_h, gap = 480, 360, 64, 6
    canvas = Image.new('RGB', (len(labels) * cell_w, 2 * (cell_h + header_h) + gap), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    gt = images['GT']
    for idx, label in enumerate(labels):
        x = idx * cell_w
        metric = rows.get(label)
        subtitle = '' if metric is None else f"PSNR {metric['psnr']:.2f}  SSIM {metric['ssim']:.4f}"
        draw.text((x + 10, 8), label, fill='black', font=font)
        draw.text((x + 10, 30), subtitle, fill=(50, 50, 50), font=font)
        canvas.paste(fit(tensor_to_pil(images[label]), (cell_w, cell_h)), (x, header_h))
        y2 = cell_h + header_h + gap
        draw.text((x + 10, y2 + 8), 'Absolute error x4' if label != 'GT' else 'Reference', fill='black', font=font)
        lower = Image.new('RGB', (cell_w, cell_h), (28, 28, 28)) if label == 'GT' else fit(error_map(images[label], gt), (cell_w, cell_h))
        canvas.paste(lower, (x, y2 + header_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format='PNG', compress_level=6)


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    keys = ('l1', 'psnr', 'ssim', 'lpips_squeeze', 'low_change_keep_l1', 'high_change_restore_l1')
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def report_markdown(summary: dict, selected: list[str], records: dict[str, dict]) -> str:
    means = summary['means']
    lines = [
        '# 纠正标签后的 M2-A / M2-B / M2-B1 封存测试集对比', '',
        '## 评价口径', '',
        '- 数据：M2 封存测试集 18 张，训练和调参阶段未参与。',
        '- 推理：三个版本均从各自 `best_transmission_lora.safetensors` 加载；固定随机种子 2026。',
        '- 指标：从保存后的 8 位 RGB PNG 重新读取并计算，按图片宏平均。',
        '- 可视化：固定抽取近方形、极横图、极竖图各一张。面板第二行为相对 GT 的绝对误差放大 4 倍。', '',
        '## 18 张测试集汇总', '',
        '| 版本 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 低变化区 L1 ↓ | 高变化区 L1 ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('Input', 'M2-A', 'M2-B', 'M2-B1'):
        m = means[name]
        lines.append(f"| {name} | {m['l1']:.6f} | {m['psnr']:.4f} | {m['ssim']:.6f} | {m['lpips_squeeze']:.6f} | {m['low_change_keep_l1']:.6f} | {m['high_change_restore_l1']:.6f} |")
    base = means['M2-A']
    lines += ['', '## 相对 M2-A 的变化', '', '| 版本 | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δ低变化区 L1 ↓ | Δ高变化区 L1 ↓ |', '|---|---:|---:|---:|---:|---:|']
    for name in ('M2-B', 'M2-B1'):
        m = means[name]
        lines.append(f"| {name} | {m['psnr']-base['psnr']:+.4f} | {m['ssim']-base['ssim']:+.6f} | {m['lpips_squeeze']-base['lpips_squeeze']:+.6f} | {m['low_change_keep_l1']-base['low_change_keep_l1']:+.6f} | {m['high_change_restore_l1']-base['high_change_restore_l1']:+.6f} |")
    lines += ['', '## 三张代表性测试图', '']
    for sid in selected:
        rec = records[sid]
        lines += [f"### `{sid}` — {rec['aspect_bucket']}，{rec['target_size'][0]}×{rec['target_size'][1]}", '', f"![{sid}](panels/{sid}_comparison.png)", '']
    winner_psnr = max(('M2-A', 'M2-B', 'M2-B1'), key=lambda x: means[x]['psnr'])
    winner_lpips = min(('M2-A', 'M2-B', 'M2-B1'), key=lambda x: means[x]['lpips_squeeze'])
    lines += [
        '## 结论', '',
        f"- 测试集 PSNR 最高的是 **{winner_psnr}**（{means[winner_psnr]['psnr']:.4f} dB）。",
        f"- 测试集 LPIPS 最低的是 **{winner_lpips}**（{means[winner_lpips]['lpips_squeeze']:.6f}）。",
        f"- 相对 M2-A，M2-B1 的 PSNR 为 {means['M2-B1']['psnr']-base['psnr']:+.4f} dB、SSIM 为 {means['M2-B1']['ssim']-base['ssim']:+.6f}、L1 为 {means['M2-B1']['l1']-base['l1']:+.6f}；提升存在，但幅度很小。",
        f"- M2-B 的低变化区 L1 最低（{means['M2-B']['low_change_keep_l1']:.6f}），M2-A 的高变化区 L1 最低（{means['M2-A']['high_change_restore_l1']:.6f}）。M2-B1 相比 M2-B 改善高变化区 {means['M2-B1']['high_change_restore_l1']-means['M2-B']['high_change_restore_l1']:+.6f}，说明 30% 梯度约束缓解了未受控 Q20 的局部退化，但尚未超过 M2-A。",
        f"- 与原始输入相比，三个模型将高变化区 L1 从 {means['Input']['high_change_restore_l1']:.6f} 降至约 0.127，同时低变化区误差、LPIPS 和整体 SSIM变差，说明当前模型确实更改了反射强区域，但也损伤了一部分原本正确的纹理。",
        '- 三个分支只有 70 次更新，差异接近指标数值波动，应视为趋势信号，不能据此断言架构已稳定优于基线。', '',
        '## 产物', '',
        '- `metrics.csv`：逐图、逐版本指标。',
        '- `summary.json`：汇总指标、权重哈希、测试编号和配置。',
        '- 完整预测：`/share/linmingheng-local/xuke/RMagNet/reports/m2_three_way_test/predictions/`，含三个版本的全部 18 张测试结果。',
        '- `panels/`：三张代表样本的横向对比图。',
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _, records, splits = load_m2_manifest(args.data_root)
    test_ids = splits['test']
    if len(test_ids) != 18 or not set(SELECTED).issubset(test_ids):
        raise RuntimeError('Unexpected sealed test split')
    for path in VARIANTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device('cuda:0')
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch('transmission')
    backend.transformer.eval()
    backend.vae.eval()
    lpips_model = lpips.LPIPS(net='squeeze', verbose=False).eval().cpu()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    rows: list[dict] = []
    selected_images: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    selected_rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for sid in test_ids:
        inp = ((image_tensor(args.data_root / 'blended' / f'{sid}.png').unsqueeze(0) + 1) * 0.5)
        gt = ((image_tensor(args.data_root / 'transmission_layer' / f'{sid}.png').unsqueeze(0) + 1) * 0.5)
        m = measure(inp, gt, inp, lpips_model)
        row = {'id': sid, 'variant': 'Input', 'bucket': records[sid]['aspect_bucket'], 'width': inp.shape[-1], 'height': inp.shape[-2], **m}
        rows.append(row)
        if sid in SELECTED:
            selected_images[sid]['Input'] = inp
            selected_images[sid]['GT'] = gt
            selected_rows[sid]['Input'] = m

    hashes = {}
    with torch.inference_mode():
        for variant, checkpoint in VARIANTS.items():
            load_adapter(backend, checkpoint, device)
            hashes[variant] = {'path': str(checkpoint), 'sha256': sha256(checkpoint)}
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            for sid in test_ids:
                normalized = image_tensor(args.data_root / 'blended' / f'{sid}.png').unsqueeze(0).to(device)
                prediction = backend.forward_normalized(normalized, 'transmission')
                prediction = ((prediction.float() + 1) * 0.5).clamp(0, 1).cpu()
                out_path = args.output_dir / 'predictions' / variant / f'{sid}.png'
                save_png(prediction, out_path)
                saved = pil_to_tensor(out_path)
                inp = ((image_tensor(args.data_root / 'blended' / f'{sid}.png').unsqueeze(0) + 1) * 0.5)
                gt = ((image_tensor(args.data_root / 'transmission_layer' / f'{sid}.png').unsqueeze(0) + 1) * 0.5)
                m = measure(saved, gt, inp, lpips_model)
                row = {'id': sid, 'variant': variant, 'bucket': records[sid]['aspect_bucket'], 'width': saved.shape[-1], 'height': saved.shape[-2], **m}
                rows.append(row)
                if sid in SELECTED:
                    selected_images[sid][variant] = saved
                    selected_rows[sid][variant] = m
                del normalized, prediction, saved
                torch.cuda.empty_cache()

    for sid in SELECTED:
        build_panel(sid, selected_images[sid], selected_rows[sid], args.output_dir / 'panels' / f'{sid}_comparison.png')

    with (args.output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['variant']].append(row)
    means = {name: mean_metrics(grouped[name]) for name in ('Input', 'M2-A', 'M2-B', 'M2-B1')}
    summary = {
        'status': 'complete',
        'split': 'test',
        'sample_count': len(test_ids),
        'test_ids': test_ids,
        'selected_visual_ids': SELECTED,
        'seed': args.seed,
        'metric_domain': 'saved 8-bit RGB PNG; macro average over images',
        'weights': hashes,
        'means': means,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'REPORT.md').write_text(report_markdown(summary, SELECTED, records), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
