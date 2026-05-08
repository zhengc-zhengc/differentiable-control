"""为某个 checkpoint 运行完整诊断流程（数据采集 + 4 张图）。

通过两个 monkey-patch 复用现有脚本，无需改它们的源码：
  1. os.path.join：当遇到 ".../results/diagnostic/mlp_instability" 路径时，
     自动追加 <subdir> → 改成 ".../results/diagnostic/mlp_instability/<subdir>"
  2. Figure.savefig：保存前递归把图里所有文字里的 '0507' 替换成 <test-label>

用法示例：
    python sim/debug/run_for_ckpt.py \\
        --subdir 0508_train_loss \\
        --test-ckpt configs/checkpoints/best_truck_trailer_error_model_train_loss_0508.pth \\
        --test-label 0508TL
"""
import os, sys, argparse, runpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(THIS_DIR)


def patch_savefig_relabeling(old_label, new_label):
    """patch Figure.savefig：保存前把整张图里所有文字的 old_label 替换成 new_label。"""
    import matplotlib.figure as _fig

    orig_savefig = _fig.Figure.savefig

    def collect_text_artists(fig):
        """递归收集 fig 里所有 Text artists。"""
        texts = []
        for ax in fig.get_axes():
            if ax.title is not None:
                texts.append(ax.title)
            for ax_lbl in [ax.xaxis.label, ax.yaxis.label]:
                texts.append(ax_lbl)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                texts.append(tick)
            leg = ax.get_legend()
            if leg is not None:
                texts.extend(leg.get_texts())
            texts.extend(list(ax.texts))
        if getattr(fig, '_suptitle', None) is not None:
            texts.append(fig._suptitle)
        # colorbar 上的 label
        for ax in fig.get_axes():
            if hasattr(ax, '_colorbar'):
                texts.append(ax._colorbar.ax.yaxis.label)
        texts.extend(list(fig.texts))
        return texts

    def patched_savefig(self, *args, **kwargs):
        for t in collect_text_artists(self):
            try:
                s = t.get_text()
                if isinstance(s, str) and old_label in s:
                    t.set_text(s.replace(old_label, new_label))
            except Exception:
                pass
        return orig_savefig(self, *args, **kwargs)

    _fig.Figure.savefig = patched_savefig


def patch_path_join(out_subdir_full):
    """patch os.path.join：碰到 'results/diagnostic/mlp_instability' 路径时，
    在它之后追加 <subdir>，相当于把读写都重定向到子目录。"""
    import os as _os
    real_join = _os.path.join
    marker_parts = ('results', 'diagnostic', 'mlp_instability')

    def patched_join(*parts):
        p = real_join(*parts)
        # 找 'results/diagnostic/mlp_instability' 这个连续片段的位置
        try:
            idx_str = (_os.sep).join(marker_parts)
            if idx_str in p:
                pos = p.index(idx_str) + len(idx_str)
                # 已经包含子目录则不改
                tail = p[pos:]
                if tail.startswith(_os.sep + os.path.basename(out_subdir_full)) \
                        or tail == _os.sep + os.path.basename(out_subdir_full) \
                        or os.path.basename(out_subdir_full) in tail.split(_os.sep):
                    return p
                return p[:pos] + _os.sep + os.path.basename(out_subdir_full) + tail
        except Exception:
            pass
        return p

    _os.path.join = patched_join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', required=True,
                        help='输出子目录名（在 results/diagnostic/mlp_instability/ 下）')
    parser.add_argument('--test-ckpt', required=True,
                        help='被测 MLP checkpoint 路径（相对 sim/ 目录）')
    parser.add_argument('--test-label', required=True,
                        help='被测 MLP 在图里的标签（如 0508TL）')
    parser.add_argument('--skip-collect', action='store_true')
    parser.add_argument('--scenarios', nargs='+', default=None)
    args = parser.parse_args()

    os.chdir(SIM_DIR)
    sys.path.insert(0, SIM_DIR)
    out_subdir_full = os.path.join(SIM_DIR, 'results', 'diagnostic',
                                    'mlp_instability', args.subdir)
    os.makedirs(out_subdir_full, exist_ok=True)

    # 让被调脚本通过环境变量看到测试 ckpt（plot_*.py 的 load_mlp / 静态扫描读它）
    ckpt_basename = os.path.basename(args.test_ckpt)
    os.environ['TEST_CKPT_NAME'] = ckpt_basename
    os.environ['TEST_LABEL'] = args.test_label
    print(f'>>> 设置 TEST_CKPT_NAME={ckpt_basename}')
    print(f'>>> 设置 TEST_LABEL={args.test_label}')

    # ===== 阶段 1：数据采集 =====
    if not args.skip_collect:
        print(f'\n>>> 阶段 1：数据采集 → {out_subdir_full}/')
        sys.argv = ['investigate_mlp_instability.py',
                    '--subdir', args.subdir,
                    '--test-ckpt', args.test_ckpt]
        if args.scenarios:
            sys.argv += ['--scenarios'] + list(args.scenarios)
        runpy.run_path(os.path.join(THIS_DIR,
                                     'investigate_mlp_instability.py'),
                       run_name='__main__')
    else:
        print(f'\n>>> 跳过阶段 1，复用 {out_subdir_full}/ 下已有数据')

    # ===== Patch：路径重定向 + 标签替换 =====
    patch_path_join(out_subdir_full)
    patch_savefig_relabeling('0507', args.test_label)

    # ===== 阶段 2 / 3 / 4 =====
    for script in ['plot_mlp_instability.py', 'plot_root_cause_story.py',
                   'plot_mlp_danger_zones.py']:
        print(f'\n>>> 阶段：{script}')
        sys.argv = [script]
        runpy.run_path(os.path.join(THIS_DIR, script), run_name='__main__')

    print(f'\n所有产物已存：{out_subdir_full}/')


if __name__ == '__main__':
    main()
