# Executive Summary: Predicting NBA Player Longevity
The goal of this predictive model is to identify whether a rookie player will have an NBA career lasting 5 years or longer (target_5yrs = 1). In the context of front-office roster optimization, false classifications carry severe financial consequences. Our model utilizes a Gaussian Naive Bayes (GNB) algorithmic framework to calculate probabilities based on on-court productivity metrics.

## Performance Metrics & Scouting Trade-offs
A baseline execution of this distribution profile typically yields distinct analytical patterns on standard rookie splits:

Precision (approx. 78%): Out of all rookies the model flags as a "safe 5-year bet," 78% actually make it. The remaining 22% are False Positives ("Busts")—players the model cleared, but who washed out early.

Recall (approx. 60-65%): Out of all rookies who genuinely go on to have durable careers, the model successfully flags 60-65% of them. The unflagged balance represents False Negatives ("Missed Talent")—underrated rotation gems the model mistakenly dismissed as short-lived.

💡 Scouting Strategy Directives: Because Gaussian Naive Bayes tends to favor high-density classes (and the league base-rate is naturally slanted toward ~62% of drafted rookies surviving 5 years via multi-year rookie scale guarantees), the model aggressively predicts survival. If your front office wants to minimize draft-pick busts, you should look for a model with higher Precision constraints. If your goal is to find undervalued diamonds in the rough late in the draft, prioritize high Recall.

## Critiquing the Naive Bayes "Independence Assumption"
The defining math behind Naive Bayes is its assumption of conditional independence—meaning it calculates the probability of each player trait completely in a vacuum, ignoring how traits interact with one another.

## The Real-World Breakdown in Basketball Stats
In sports analytics, this assumption is fundamentally broken. On-court statistics are deeply intertwined:

total_points is a direct mathematical product of games played (gp) and points per game (pts).

tov (Turnovers) is heavily correlated with high ast (Assists) and high usage rates; pass-heavy playmakers naturally carry higher turnover risks.

reb (Rebounds) and blk (Blocks) are naturally heavily correlated based on a player's physical height/positioning (frontcourt vs. backcourt).

## Impact on Scouting Validity
Because Naive Bayes treats highly correlated stats as independent pieces of evidence, it essentially double-counts identical underlying talent traits. For example, if a player scores heavily, the model receives positive signals from pts, fg, and total_points separately, artificially spiking its confidence score. This creates highly exaggerated probability distributions (predictions pushed too close to 100% or 0%).

## Key Predictive Features Driving the Model
Based on the mean separation thresholds between structural survivors and non-survivors, the features that swing the model's confidence the most include:

total_points / efficiency: Reflective of volume-scoring utility and high offensive output right out of the gate.

ft / fg (Percentages): Traditional efficiency metrics cleanly separate structural rotation pieces from raw prospects.

reb and ast: Volume tracking metrics indicate overall engagement and foundational baseline performance.

## Actionable Strategic Recommendations
Do Not Use as an Isolated Decision Maker: Because the model suffers from extreme structural confidence inflation (due to the independence flaw), it should never be given final draft-board authority.

Integrate as a First-Stage Filtering Screen: Use this model during early draft preparation to ingest data across hundreds of prospects instantly. It functions perfectly as an automated flagger to highlight players whose baseline productivity profiles mathematically match historical 5-year survivors.

Combine with Contextual Modeling: Pair this probabilistic tool with modern tree-based ensembles (like XGBoost or Random Forests) that inherently capture non-linear relationships (e.g., how high turnovers are perfectly acceptable if accompanied by high assist rates).
