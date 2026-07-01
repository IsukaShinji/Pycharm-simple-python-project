import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rankdata
from scipy.special import softmax
from multiprocessing import Pool, cpu_count
import warnings
import os

# Filter warnings
warnings.filterwarnings('ignore')

# Set plotting style (Times New Roman for academic standards)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("deep")


# ==========================================
# 1. Global Helper Functions (Logic Core)
# ==========================================

def get_season_rule(season):
    """
    Determine the rule logic based on the specific season era.
    S1-S2: Rank (Sum of Ranks)
    S3-S27: Percent (Sum of Percentages)
    S28-S34: Rank + Judges' Save (Bottom 2 by Rank, then Judges pick)
    """
    if season in [1, 2]:
        return 'Rank'
    elif 3 <= season <= 27:
        return 'Percent'
    elif 28 <= season <= 34:
        return 'Rank_Save'
    return 'Percent'


def get_nominal_votes_pool(season, week):
    """
    Estimate nominal total votes pool to convert shares to integers.
    Assumption: Votes grow with season popularity and intra-season excitement.
    """
    base_votes = 1_000_000
    season_factor = 1.0 + (season * 0.05)

    if week <= 7:
        week_factor = 1.0 + (week * 0.02)
    else:
        week_factor = 1.15 + ((week - 7) * 0.1)

    return int(base_votes * season_factor * week_factor)


def _calculate_rank_save_prob(p_vec, j_scores, elim_indices, temp=5.0):
    """
    Complex Logic for S28+: Rank + Judges' Save.
    Probability = P(In Bottom 2) * P(Not Saved by Judges | In Bottom 2)
    """
    n = len(p_vec)
    # 1. Calculate Rank Sum
    r_j = rankdata(-j_scores, method='min')  # Higher score -> Lower rank num (1 is best)
    r_f = rankdata(-p_vec, method='min')
    total_rank = r_j + r_f

    # 2. Softmax to determine probability of being in Bottom 2
    # We want high rank sums (bad performance) to have high prob.
    # Scores for softmax = total_rank * temp
    scores = total_rank * temp
    probs_b2 = softmax(scores - np.max(scores))  # Prob of being the "worst"

    # 3. Calculate "Save Probability"
    # If in Bottom 2, judges save the one with higher Judge Score.
    # We approximate the "Rival" in B2 as the person with the next highest rank sum.

    energy_penalty = 0.0

    for idx in elim_indices:
        # P(In Danger)
        prob_danger = probs_b2[idx]

        # Find likely rival (Max rank sum among others)
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        rival_idx = np.argmax(total_rank * mask)

        # Judge Decision Model: Sigmoid based on score difference
        # If MyScore < RivalScore, I am likely eliminated.
        # If MyScore > RivalScore, I am likely saved.
        score_diff = j_scores[idx] - j_scores[rival_idx]

        # Sigmoid: P(Elim | B2) = 1 / (1 + exp(k * score_diff))
        # k is the "Judge Rationality" factor.
        k_judge = 2.0
        prob_elim_given_b2 = 1.0 / (1.0 + np.exp(k_judge * score_diff))

        # Total Prob of Elimination
        total_prob = prob_danger * prob_elim_given_b2

        # Avoid log(0)
        energy_penalty -= np.log(total_prob + 1e-10)

    return energy_penalty


def calculate_energy(p_vec, j_scores, rule, actual_elim_names, contestants):
    """
    Calculate Energy (Negative Log-Likelihood) for MCMC.
    Formula: E = -ln( P(Actual Elimination | Fan Votes) )
    Uses Soft-Likelihood with Temperature coefficients.
    """
    elim_indices = [i for i, name in enumerate(contestants) if name in actual_elim_names]
    if not elim_indices:
        return 0.0

    energy = 0.0
    temp_percent = 50.0  # High temp for sharp boundaries
    temp_rank = 10.0

    if rule == 'Percent':
        # Logic: Lowest Total Percent is eliminated.
        j_share = j_scores / (j_scores.sum() + 1e-9)
        total_share = j_share + p_vec

        # We want to maximize P(Eliminated Person is Min).
        # Softmax on negative scores -> Min gets highest prob.
        scores = -total_share * temp_percent
        probs = softmax(scores - np.max(scores))

        for idx in elim_indices:
            energy -= np.log(probs[idx] + 1e-10)

    elif rule == 'Rank':
        # Logic: Highest Rank Sum is eliminated.
        r_j = rankdata(-j_scores, method='min')
        r_f = rankdata(-p_vec, method='min')
        total_rank = r_j + r_f

        # Softmax on positive rank sums -> Max gets highest prob.
        scores = total_rank * temp_rank
        probs = softmax(scores - np.max(scores))

        for idx in elim_indices:
            energy -= np.log(probs[idx] + 1e-10)

    elif rule == 'Rank_Save':
        # Delegate to specialized logic
        energy = _calculate_rank_save_prob(p_vec, j_scores, elim_indices)

    return energy


def run_mcmc_worker(args):
    """
    Worker function for MCMC.
    """
    season, week, contestants, j_scores, actual_elim, n_samples, burn_in = args
    rule = get_season_rule(season)
    n = len(contestants)

    # Initialization: Random Dirichlet (Sum=1)
    current_p = np.random.dirichlet(np.ones(n))
    actual_elim_set = set(actual_elim)

    # Initial Energy
    current_energy = calculate_energy(current_p, j_scores, rule, actual_elim_set, contestants)

    samples = []
    proposal_width = 0.03  # Perturbation scale

    for i in range(n_samples + burn_in):
        # Proposal: Swap Delta
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        delta = np.random.normal(0, proposal_width)

        # Bounds check
        if (current_p[idx1] + delta < 0) or (current_p[idx1] + delta > 1) or \
                (current_p[idx2] - delta < 0) or (current_p[idx2] - delta > 1):
            continue

        proposal = current_p.copy()
        proposal[idx1] += delta
        proposal[idx2] -= delta

        new_energy = calculate_energy(proposal, j_scores, rule, actual_elim_set, contestants)

        # Metropolis Accept/Reject
        if np.random.rand() < np.exp(current_energy - new_energy):
            current_p = proposal
            current_energy = new_energy

        if i >= burn_in:
            samples.append(current_p.copy())

    return (season, week, np.array(samples) if samples else np.zeros((0, n)))


# ==========================================
# 2. Main Solver Class (Three-Stage Progressive Inference)
# ==========================================

class DWTSSolver:
    def __init__(self, train_path, test_path, raw_path):
        self.train_path = train_path
        self.test_path = test_path
        self.raw_path = raw_path
        self.long_df = None
        self.model = None

    def load_data(self):
        print("1. Loading Data...")
        try:
            df_train = pd.read_excel(self.train_path)
            df_test = pd.read_excel(self.test_path)
        except Exception:
            # Fallback for CSV if extension mismatches
            df_train = pd.read_csv(self.train_path)
            df_test = pd.read_csv(self.test_path)

        df_train['is_train'] = 1
        df_test['is_train'] = 0
        self.full_data = pd.concat([df_train, df_test], ignore_index=True)
        print(f"   Loaded {len(self.full_data)} records.")

    def prepare_long_format(self):
        print("2. Preprocessing Data...")
        long_records = []

        for idx, row in self.full_data.iterrows():
            season = row['season']
            name = row['celebrity_name']
            elim_week = row['eliminated_week']
            final_status = row['final_status']
            is_train = row['is_train']

            # Identify active weeks
            for w in range(1, 13):  # Scan up to week 12
                col = f'week{w}_judge_total'
                if col not in row or pd.isna(row[col]) or row[col] == 0:
                    continue

                is_active = False
                if final_status != 0:  # Finalist
                    is_active = True
                else:  # Eliminated
                    if w <= elim_week:
                        is_active = True

                if is_active:
                    elim_this_week = 1 if (final_status == 0 and w == elim_week) else 0
                    long_records.append({
                        'season': season,
                        'week': w,
                        'celebrity_name': name,
                        'judge_total': row[col],
                        'eliminated_this_week': elim_this_week,
                        'age': row['celebrity_age_during_season'],
                        'industry': row['celebrity_industry'],
                        'partner': row['ballroom_partner'],
                        'is_train': is_train
                    })
        self.long_df = pd.DataFrame(long_records)
        print(f"   Prepared {len(self.long_df)} weekly records.")

    def stage1_inference(self):
        """
        [Stage 1: Inference]
        Run MCMC Back-Inference ONLY on Training Set.
        Goal: Reconstruct the 'Latent' Fan Votes that explain history.
        """
        print("3. [Stage 1] Inference: MCMC Back-Inference (Training Set Only)...")
        train_df = self.long_df[self.long_df['is_train'] == 1]
        grouped = train_df.groupby(['season', 'week'])

        tasks = []
        meta = []

        # MCMC Configuration
        N_SAMPLES = 3000
        BURN_IN = 1000

        for (season, week), group in grouped:
            contestants = group['celebrity_name'].values
            j_scores = group['judge_total'].values
            actual_elim = group[group['eliminated_this_week'] == 1]['celebrity_name'].values

            if len(contestants) > 1:
                tasks.append((season, week, contestants, j_scores, actual_elim, N_SAMPLES, BURN_IN))
                meta.append(group.index)

        # Parallel Execution
        n_cores = min(16, max(1, int(cpu_count() * 0.9)))
        print(f"   Using {n_cores} cores for {len(tasks)} tasks.")

        results_map = {}
        with Pool(n_cores) as pool:
            results = pool.map(run_mcmc_worker, tasks)

        for res in results:
            s, w, samples = res
            results_map[(s, w)] = samples

        # Store results
        est_shares = np.zeros(len(self.long_df))
        est_shares[:] = np.nan
        est_cert = np.zeros(len(self.long_df))
        est_votes = np.zeros(len(self.long_df))

        for i, indices in enumerate(meta):
            s, w = tasks[i][0], tasks[i][1]
            samples = results_map.get((s, w))

            if samples.shape[0] > 0:
                means = samples.mean(axis=0)
                stds = samples.std(axis=0)
                pool_size = get_nominal_votes_pool(s, w)

                for j, idx in enumerate(indices):
                    p = means[j]
                    std = stds[j]
                    est_shares[idx] = p

                    # [Logic] Certainty Score based on CV (Coefficient of Variation)
                    # High Mean + Low Std = High Certainty
                    # Low Mean + Low Std = High Certainty (We know they are unpopular)
                    # Certainty = 1 / (1 + CV) -> Maps to [0, 1] roughly
                    if p > 1e-9:
                        cv = std / p
                        est_cert[idx] = 1.0 / (1.0 + cv)
                    else:
                        est_cert[idx] = 1.0  # Certain it's zero

                    est_votes[idx] = int(p * pool_size)

        self.long_df['mcmc_fan_share'] = est_shares
        self.long_df['mcmc_fan_votes'] = est_votes
        self.long_df['certainty'] = est_cert

    def stage2_learning(self):
        """
        [Stage 2: Learning]
        Train a Random Forest to learn the relationship:
        Features (Age, Industry, JudgeScore) -> Target (Inferred Fan Share)
        """
        print("4. [Stage 2] Learning: Feature Association (Random Forest)...")

        # Use only rows where MCMC provided a target
        train_data = self.long_df[(self.long_df['is_train'] == 1) & (self.long_df['mcmc_fan_share'].notna())]

        # Feature Engineering
        X_train = train_data[['judge_total', 'age']]
        X_train = pd.concat([X_train, pd.get_dummies(train_data['industry'], prefix='Ind')], axis=1)

        # Save columns for alignment
        self.feature_cols = X_train.columns
        y_train = train_data['mcmc_fan_share']

        # Model Training
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.model = rf

        # Calculate Feature Importance
        self.imps = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})

        # Predict on Test Set (Blind Prediction)
        test_data = self.long_df[self.long_df['is_train'] == 0]
        X_test = test_data[['judge_total', 'age']]
        X_test = pd.concat([X_test, pd.get_dummies(test_data['industry'], prefix='Ind')], axis=1)

        # Align Columns
        for col in self.feature_cols:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self.feature_cols]

        # Predict
        y_pred = rf.predict(X_test)
        self.long_df.loc[test_data.index, 'predicted_fan_share'] = y_pred

    def stage3_validation(self):
        """
        [Stage 3: Validation]
        Generalization Check.
        Use Predicted Fan Shares + Season Rules -> Simulate Elimination -> Check Match.
        """
        print("5. [Stage 3] Validation: Generalization Check (Out-of-Sample)...")
        hits = 0
        total = 0

        test_df = self.long_df[self.long_df['is_train'] == 0]

        for (s, w), g in test_df.groupby(['season', 'week']):
            if g['eliminated_this_week'].sum() == 0: continue
            if len(g) < 2: continue

            rule = get_season_rule(s)
            names = g['celebrity_name'].values
            j_s = g['judge_total'].values
            # CRITICAL: Use PREDICTED share from RF, not MCMC
            f_s = g['predicted_fan_share'].values
            actual = g[g['eliminated_this_week'] == 1]['celebrity_name'].values

            pred = None
            try:
                if rule == 'Percent':
                    j_p = j_s / (j_s.sum() + 1e-9)
                    pred = names[np.argmin(j_p + f_s)]

                elif rule == 'Rank':
                    r_j = rankdata(-j_s, method='min')
                    r_f = rankdata(-f_s, method='min')
                    pred = names[np.argmax(r_j + r_f)]

                elif rule == 'Rank_Save':
                    r_j = rankdata(-j_s, method='min')
                    r_f = rankdata(-f_s, method='min')
                    tot = r_j + r_f

                    # Bottom 2 Logic
                    if len(tot) >= 2:
                        # Tie-breaking for B2: if rank sums equal, use fan rank
                        b2_indices = np.lexsort((-f_s, -tot))[:2]  # Sort by total desc, then fan desc
                        # Note: argsort default is ascending, we want descending for "Worst"
                        # Actually simpler: just pick top 2 totals
                        b2 = np.argsort(-tot)[:2]

                        # Judge Save: Judges save the one with HIGHER judge score
                        # So the one with LOWER judge score is ELIMINATED
                        if j_s[b2[0]] < j_s[b2[1]]:
                            pred = names[b2[0]]
                        else:
                            pred = names[b2[1]]
                    else:
                        pred = names[np.argmax(tot)]

            except Exception:
                continue

            if pred in actual:
                hits += 1
            total += 1

        acc = hits / total if total > 0 else 0
        print(f"   Test Set Consistency (Hit Rate): {acc:.2%}")
        return acc

    def visualize(self, accuracy):
        print("6. Visualizing Results...")

        # 1. Bubble Plot: Certainty Paradox
        # Shows how "Certainty" relates to the conflict between Judge Score and Fan Share
        plt.figure(figsize=(10, 6))
        # Plot Train set (Inferred)
        plot_df = self.long_df[(self.long_df['is_train'] == 1) & (self.long_df['mcmc_fan_share'].notna())].sample(
            n=min(500, len(self.long_df)), random_state=42)

        colors = plot_df['eliminated_this_week'].map({1: 'red', 0: 'cornflowerblue'})
        sizes = plot_df['certainty'] * 150 + 20

        plt.scatter(plot_df['judge_total'], plot_df['mcmc_fan_share'],
                    s=sizes, c=colors, alpha=0.6, edgecolors='w', linewidth=0.5)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Eliminated'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', label='Safe'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Size ~ Certainty', markersize=5)
        ]
        plt.legend(handles=legend_elements, loc='best')
        plt.title('The Certainty Paradox: Inferred Fan Share vs Judge Score', fontsize=14)
        plt.xlabel('Judge Total Score')
        plt.ylabel('Inferred Fan Vote Share')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('Q1_Certainty_Paradox.png', dpi=300)
        plt.show()

        # 2. Feature Importance
        plt.figure(figsize=(10, 6))
        top_imps = self.imps.sort_values('Importance', ascending=False).head(10)
        sns.barplot(data=top_imps, x='Importance', y='Feature', palette='viridis')
        plt.title('What Drives Fan Votes? (Learned Feature Importance)', fontsize=14)
        plt.tight_layout()
        plt.savefig('Q1_Feature_Drivers.png', dpi=300)
        plt.show()

        # 3. Validation Performance (Confusion Matrix style or just Bar)
        # Simple text annotation
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, f"Out-of-Sample Hit Rate: {accuracy:.2%}",
                 fontsize=20, ha='center', va='center')
        plt.axis('off')
        plt.title("Model Generalization Capability")
        plt.tight_layout()
        plt.savefig('Q1_Validation_Score.png', dpi=300)
        plt.show()

    def run(self):
        self.load_data()
        self.prepare_long_format()

        # Stage 1: Inference
        self.stage1_inference()

        # Stage 2: Learning
        self.stage2_learning()

        # Stage 3: Validation
        acc = self.stage3_validation()

        # Visualization
        self.visualize(acc)

        # Save final output
        self.long_df.to_csv('Q1_Three_Stage_Results.csv', index=False)
        print("Done. Detailed results saved to Q1_Three_Stage_Results.csv")


if __name__ == "__main__":
    # Define paths
    train_file = r"C:\Users\21165\Desktop\2026_MCM-ICM_Problems\train_set_weekly.xlsx"
    test_file = r"C:\Users\21165\Desktop\2026_MCM-ICM_Problems\test_set_weekly.xlsx"
    raw_file = r"C:\Users\21165\Desktop\2026_MCM-ICM_Problems\2026_MCM_Problem_C_Data.xlsx"

    solver = DWTSSolver(train_file, test_file, raw_file)
    solver.run()