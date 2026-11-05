# Boréal Marché — executive summary

Eraldo Ibrushi · Vanier capstone · 5 November 2026 · English

## Purpose and results

Boréal Marché combines three tasks in one Streamlit/FastAPI demonstration: predict whether a customer buys within 90 days; suggest products that co-occur in purchase baskets; and answer policy questions through retrieval and typed tools. Its public historical data comes from UK giftware retail, while the Québec grocery business is fictional. The original monetary values remain GBP.

Head A’s current calibrated Random Forest uses 12 serving features from 25 engineered offline. On the reused September test period, average precision is 0.760 versus 0.671 for recency ranking; ROC-AUC is 0.786 and Brier is 0.1940. A 1,000-draw customer bootstrap gives AP interval 0.742–0.775. This measures historical purchase ranking, not coupon effectiveness. The 20.5% operating threshold maximises an assumed June utility; coupon cost 5 and margin 25 remain scenarios. Equal-count recency nearly matches recall: 81.93% versus 82.42%.

Head B exports 3,567 FP-Growth association rules, checked against Apriori. Temporal half-basket Hit@5 is 48.7% versus 17.2% popularity. Rules fire for 83.7% of evaluated baskets, but suggestions cover only 7.7% of the catalogue. These measures have different denominators. Apriori was faster in the measured bounded mining comparison; indexed serving is separate from mining.

Head C offers regex, OpenAI and Ollama providers with multilingual retrieval and four typed tools. Routing scores are 97.5%, 95% and 92.5% on 40 development questions. Historical written-answer review marked only 11/18 OpenAI and 3/18 Ollama answers correct, using one AI rater. French answer correctness requires independent review.

## Delivery and next decisions

Local verification reports 428 tests passing, 90% source/API coverage, 16 HTTP checks and seven Streamlit interaction checks. Thirteen unnecessary model inputs were removed; thin-history fairness gaps remain. Human review and no automatic coupon dispatch are safeguards, not evidence of statistical parity or legal compliance.

The local prototype and submission materials are prepared. GitHub authentication, AWS credentials and a spending cap are still needed for actual CI/cloud runs, hosting and release evidence. Video and owner-verified reflection also remain. Next scientific work is a coupon/control experiment, a future-period holdout and independent bilingual answer evaluation.

Evidence: MODEL_CARD.md; retention_diagnostics.json; recommendation_metrics.json; answer_correctness.json; completion logs; docs/AI_USE.md. The earlier model is archived and test-period reuse is disclosed.

<!-- PAGEBREAK -->

# Boréal Marché — résumé exécutif

Eraldo Ibrushi · Projet de synthèse Vanier · 5 novembre 2026 · Français

## Objectif et résultats

Boréal Marché réunit trois fonctions dans une démonstration Streamlit/FastAPI : prévoir un achat dans les 90 jours, proposer des produits associés au panier et répondre aux questions de politique avec des sources et des outils typés. Les données historiques publiques proviennent d’un commerce britannique de cadeaux; l’entreprise québécoise est fictive. Les montants d’origine restent en GBP.

Le modèle A est une forêt aléatoire calibrée utilisant 12 variables en service parmi 25 variables calculées hors ligne. Sur la période de test de septembre réutilisée, la précision moyenne est de 0,760 contre 0,671 pour le classement par récence; l’aire ROC est de 0,786 et le score de Brier de 0,1940. L’intervalle bootstrap de la précision moyenne est de 0,742 à 0,775. Le seuil de 20,5 % optimise une utilité supposée en juin. Le coût du coupon (5) et la marge (25) sont des hypothèses, sans effet causal mesuré. À effectif sélectionné égal, le rappel atteint 82,42 %, contre 81,93 % pour la récence.

Le modèle B exporte 3 567 règles FP-Growth, comparées à Apriori. En masquant environ la moitié du panier, le taux de réussite parmi cinq suggestions atteint 48,7 %, contre 17,2 % pour la popularité. Une règle s’applique dans 83,7 % des paniers évalués, mais les suggestions couvrent seulement 7,7 % du catalogue. Apriori était plus rapide dans la comparaison mesurée.

L’assistant C propose regex, OpenAI et Ollama, avec recherche multilingue et quatre outils typés. Les taux de bon routage sont de 97,5 %, 95 % et 92,5 % sur 40 questions de développement. Une évaluation historique par un seul évaluateur IA juge correctes 11 réponses sur 18 pour OpenAI et 3 sur 18 pour Ollama. Le routage correct ne garantit pas une réponse exacte en français.

## Livraison et prochaines décisions

Les vérifications locales comprennent 428 tests réussis, une couverture de 90 %, 16 contrôles HTTP et sept interactions Streamlit/API. Treize variables inutiles ont été retirées du modèle. Des écarts de rappel persistent pour les historiques courts. La révision humaine et l’absence d’envoi automatique de coupons ne prouvent ni l’équité statistique ni la conformité juridique.

Les identifiants GitHub et AWS ainsi qu’un plafond de dépenses sont nécessaires pour les exécutions cloud, l’hébergement et les preuves de livraison. La vidéo et la réflexion personnelle vérifiée restent à finaliser. Les priorités scientifiques sont une expérience coupon/témoin, une nouvelle période de test et une évaluation bilingue indépendante.

Sources : rapports locaux, journal de validation et docs/AI_USE.md. L’ancien modèle est archivé; la réutilisation du test est déclarée.
