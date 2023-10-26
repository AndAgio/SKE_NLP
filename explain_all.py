import subprocess


def run_explanations_from_scratch():
    datasets = ['sms', 'trec', 'youtube', 'moral_alm_false', 'moral_baltimore_false', 'moral_blm_false', 'moral_election_false', 'moral_metoo_false', 'moral_sandy_false', 'moral_together_false']
    loc_explainers = ['gi', 'gs', 'lat', 'lrp', 'shap', 'hess', 'lime']
    aggregators = ['abs-avg', 'avg', 'abs-sum', 'sum']
    ks = [10, 50, 100, 150, 200, 250]
    for dataset in datasets:
        for loc_exp in loc_explainers:
            available_file = 0
            for aggregator in aggregators:
                for k in ks:
                    command = 'python explain.py --train_dataset="{}" --test_dataset="{}" --model="bert" --local_exp_file_available={} --local_explainer="{}" --aggregator="{}_{}" --global_explainer="cart" --run_mode="glob" --device=1 --out_folder="outs"'.format(dataset, dataset, available_file, loc_exp, aggregator, k)
                    print('Running command: {}'.format(command))
                    subprocess.run([command], shell=True)
                    available_file = 1


if __name__ == '__main__':
    run_explanations_from_scratch()
