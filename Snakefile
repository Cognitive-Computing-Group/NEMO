configfile: 'config.yaml'

subjects = glob_wildcards('data/nemo-bids/{subject}/nirs').subject
include_events = [
	"empe",
	"afim"
]
clf_methods = [
	'ind',
	'com'
]
tasks = [
	'4_class',
	'b_pn',
	'b_eb',
	'b_pene',
	'b_pbnb',
]

# subjects = subjects[:3] # for visualizing the DAG		

rule all:
	input:
		subject_specific_accuracies=expand('results/paper_results/sub_scores/{include_events}-4_class-{clf_method}-best.png', include_events=include_events, clf_method=clf_methods),
		val_aro_plot='results/paper_results/val_aro/val_aro_plot.png',
		score_tables=expand('results/paper_results/tables/{include_events}_{clf_method}_score_table.tex', include_events=include_events, clf_method=clf_methods),
		response_plots='results/paper_results/response_plots/colorbar.png',

rule load_epochs:
	input:
		script='scripts/load_bids.py',
		bids='data/nemo-bids/{subject}/nirs/{subject}_task-{include_events}_nirs.snirf'
	output:
		'processed_data/epochs/{subject}_task-{include_events}_epo.fif'
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script} -s {wildcards.subject} -e {wildcards.include_events}'

rule create_dataset:
	input:
		script='scripts/create_dataset.py',
		epochs=lambda wildcards: expand(f'processed_data/epochs/{{subject}}_task-{wildcards.include_events}_epo.fif', subject=subjects),
	output:
		X='processed_data/classification_datasets/{include_events}-{task}-MV-3-hbo/X.pkl',
		y='processed_data/classification_datasets/{include_events}-{task}-MV-3-hbo/y.pkl',
		epoch_ids='processed_data/classification_datasets/{include_events}-{task}-MV-3-hbo/epoch_ids.pkl'
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script} -e {wildcards.include_events} -t {wildcards.task}'

rule run_clf:
	input:
		script='scripts/paper_results/run_clf.py',
		X='processed_data/classification_datasets/{include_events}-{task}-MV-3-hbo/X.pkl',
		y='processed_data/classification_datasets/{include_events}-{task}-MV-3-hbo/y.pkl',
		epoch_ids='processed_data/classification_datasets/{include_events}-{task}-MV-3-hbo/epoch_ids.pkl'
	output:
		'processed_data/clf_scores/{include_events}/{clf_method}/{task}/sdf.csv'
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script} -t {wildcards.task} -m {wildcards.clf_method} -e {wildcards.include_events} --save'


rule val_aro_plot:
	input:
		script='scripts/paper_results/val_aro_plot.py',
		empe_epochs=expand('processed_data/epochs/{subject}_task-empe_epo.fif', subject=subjects),
	output:
		'results/paper_results/val_aro/val_aro_plot.png'
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script}'

rule score_table:
	input:
		script='scripts/paper_results/score_table.py',
		clf_scores=lambda wildcards: expand(f'processed_data/clf_scores/{wildcards.include_events}/{wildcards.clf_method}/{{task}}/sdf.csv', task=tasks),
	output:
		'results/paper_results/tables/{include_events}_{clf_method}_score_table.tex'
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script} -e {wildcards.include_events} -m {wildcards.clf_method} > {output}'


rule subject_specific_accuracies:
	input:
		script='scripts/paper_results/subject_specific_accuracies.py',
		clf_scores=expand('processed_data/clf_scores/{include_events}/{clf_method}/4_class/sdf.csv', include_events=include_events, clf_method=clf_methods)
	output:
		expand('results/paper_results/sub_scores/{include_events}-4_class-{clf_method}-best.png', include_events=include_events, clf_method=clf_methods)
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script}'

rule response_plots:
	input:
		script='scripts/paper_results/response_plots.py',
		epochs=expand('processed_data/epochs/{subject}_task-{include_events}_epo.fif', subject=subjects, include_events=include_events)
	output:
		'results/paper_results/response_plots/colorbar.png',
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script}'

rule permutation_test:
	input:
		script='scripts/paper_results/permutation_tests.py',
		clf_scores=lambda wildcards: expand(f'processed_data/clf_scores/{wildcards.include_events}/{wildcards.clf_method}/{{task}}/sdf.csv', task=tasks),
	output:
		'results/paper_results/permutation_test_tables/{include_events}_{clf_method}_permutation_test.tex'
	conda:
		'snakemake_env.yaml'
	shell:
		'python {input.script} -e {wildcards.include_events} -m {wildcards.clf_method} > {output}'

rule permutation_tests:
	input:
		permutation_test=expand('results/paper_results/permutation_test_tables/{include_events}_{clf_method}_permutation_test.tex', include_events=include_events, clf_method=clf_methods)
