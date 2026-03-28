"""
Evaluation and analysis pipeline for synthetic graph benchmark experiments.

Modules:
    build_metadata_tables       -- Build master experiment tables from index CSVs
    generate_feature_informativeness -- Generate synthetic node features at varying informativeness
    make_transductive_splits    -- Create 70/15/15 train/validation/test node splits
    run_structural_noise        -- Experiment 1: structural-noise sweep
    run_feature_informativeness -- Experiment 2: feature-informativeness sweep
    summarize_results           -- Produce graph-level and condition-level summary CSVs
    plot_results                -- Generate publication-ready figures
"""
