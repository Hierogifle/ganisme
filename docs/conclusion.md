# GANisme — Conclusion du projet

> Projet mené seul. Les chiffres cités sont ceux des notebooks du dépôt ; sauf mention contraire, les scores sont
> calculés sur autant d'images générées qu'il y a d'images d'entraînement (3 448 portraits, 2 714 paysages).

---

## 1. Ce qui a été accompli

Je suis parti d'un catalogue brut de 52 867 œuvres et j'arrive à quatre générateurs de peintures évalués, utilisables
dans une application.

| Étape | Résultat |
|---|---|
| Veille | 15 questions traitées, puis confrontées à mes propres mesures (question 7) |
| Analyse du catalogue | 32 438 peintures identifiées ; dates, dimensions, écoles et genres nettoyés et analysés |
| Collecte | 32 437 images téléchargées (une seule introuvable) |
| Jeux d'entraînement | 24 860 peintures après filtrage ; 3 832 portraits et 3 016 paysages au bon format, dont 10 % réservés au test |
| Métriques | FID, KID, précision, rappel, densité, couverture, mémorisation — testées sur de faux générateurs avant tout entraînement |
| Modèles | 14 entraînements complets et une recherche Optuna de 30 essais |
| Application | interface Streamlit : génération, interpolation entre deux peintures, fiche de chaque modèle |

**Les quatre modèles retenus**

| Modèle | Taille | FID | Précision | Rappel | Copies du jeu d'entraînement |
|---|---|---|---|---|---|
| Portraits | 64×80 | 54,5 | 0,62 | 0,28 | 0,0 % |
| Portraits | 128×160 | 71,1 | 0,66 | 0,10 | 0,0 % |
| Paysages | 64×48 | 42,3 | 0,86 | 0,20 | 0,2 % |
| Paysages | 128×96 | 61,9 | 0,87 | 0,07 | 0,0 % |

Chaque ligne a sa propre échelle : deux FID ne se comparent qu'à genre, résolution et nombre d'images identiques.
Pour comparer une résolution à l'autre, je réduis les images 128 pixels à 64 pixels (section 2).

---

## 2. Les quatre résultats que je retiens

**1. L'équilibre entre les deux réseaux compte plus que l'architecture.** Les trois premiers modèles, réglés à la
main, illustrent chacun un déséquilibre : discriminateur trop fort (DCGAN, FID 130), entraînement instable
(DCGAN + DiffAugment, 116 au mieux puis 340 à la dernière époque), discriminateur trop faible (normalisation
spectrale + perte hinge, 198). Avec la **même architecture**, la recherche Optuna fait passer le FID de 130 à 54,5.
Les leviers décisifs sont la taille de lot (64 au lieu de 128), deux mises à jour du discriminateur pour une du
générateur, et β₁ = 0 dans Adam.

**2. Ce gain est reproductible.** Le meilleur essai d'une recherche est toujours en partie chanceux. J'ai donc
réentraîné les deux meilleurs réglages sur trois graines et 600 époques : le réglage retenu obtient 56,0 / 54,5 / 59,0
(moyenne 56,5), l'autre 60,2 / 60,8 / 71,8. La normalisation spectrale n'améliore pas le meilleur score, elle rend
l'entraînement plus régulier d'une graine à l'autre.

**3. Le réglage se transfère.** Repris tel quel, sans nouvelle recherche :
- sur un autre genre, il fait passer les paysages de 79,8 (DCGAN de base) à 42,3 ;
- sur une résolution double, il donne des modèles meilleurs **même jugés en 64 pixels** : 41,1 contre 54,5 pour les
  portraits, 37,7 contre 42,3 pour les paysages (images 128 pixels réduites à 64).

**4. Les modèles ne recopient pas leurs données.** Les images générées sont à peu près à la même distance du jeu
d'entraînement qu'une vraie peinture jamais vue (rapport de 0,96 à 1,11 selon le modèle ; une copie donnerait 0), et
la part d'images suspectes de copie reste entre 0 et 0,2 %.

---

## 3. Les difficultés rencontrées

**Préparer les images.** Le recadrage carré centré, habituel pour les GAN, coupait les têtes des portraits et
déformait les paysages. J'ai choisi un format par genre (4:5 pour les portraits, 4:3 pour les paysages) et écarté les
peintures qui perdaient plus de 25 % de leur surface au recadrage. C'est un écart assumé par rapport au guide fourni.

**Faire confiance aux métriques.** C'est la difficulté qui m'a le plus appris. L'Inception Score ne distingue pas une
vraie peinture d'une peinture floutée. Le FID dépend du nombre d'images : une copie parfaite du réel obtient 0 sur
3 448 images et 49 sur 384. Le KID laisse passer un générateur réduit à 20 images. J'ai moi-même commis l'erreur de
comparer un FID calculé sur 3 448 images à une référence calculée sur 384, avant de la corriger en ajoutant un FID à
effectif égal.

**Stabiliser l'entraînement.** Les pertes ne disent rien de la qualité des images, et un modèle peut s'effondrer en
quelques époques (FID de 116 à 340 sur le modèle 2). Seul un suivi régulier par métriques, avec conservation
du meilleur point, permet de s'en apercevoir et de ne pas perdre le travail.

**Tenir sur un ordinateur portable.** Avec 8 Go de mémoire graphique et des calculs de plusieurs heures, j'ai dû
rendre tous les entraînements interruptibles : l'état complet est sauvegardé à chaque évaluation et la même commande
reprend là où le calcul s'est arrêté. Mes estimations de durée se sont aussi révélées fausses une fois (5,6 s par
époque prévues, 13,8 mesurées en 128×160).

---

## 4. Les limites

- **Les visages sont souvent déformés.** En 64×80, un visage occupe une quinzaine de pixels ; le passage en 128×160
  donne des images plus détaillées, mais les visages restent le point faible.
- **La diversité reste loin du réel.** Le rappel du meilleur modèle est de 0,28, contre 0,79 pour de vraies
  peintures jamais vues (valeur mesurée sur 384 images). À effectif égal, son FID est de 97,5 contre 57,6 pour le
  réel : les images générées se distinguent encore nettement de vraies peintures.
- **La convergence n'est pas atteinte.** Plusieurs modèles progressaient encore à 600 époques.
- **Peu de répétitions.** Seul le réglage des portraits 64×80 a été entraîné sur trois graines ; les modèles
  128 pixels et les paysages reposent sur un seul entraînement chacun, et aucune recherche d'hyperparamètres ne leur
  a été consacrée.
- **Le corpus est biaisé.** Peinture européenne du XVᵉ au XIXᵉ siècle, dominée par les écoles italienne, française,
  hollandaise et flamande : les images générées héritent de ce biais.
- **Les métriques reposent sur Inception-v3**, un réseau entraîné sur des photographies. Le test sur de faux
  générateurs valide le FID sur mes données, mais cela reste une mesure indirecte ; je n'ai pas mené d'évaluation
  humaine.

---

## 5. Ce que j'en retiens

1. **Mesurer avant d'entraîner.** Construire l'échelle de référence en premier a rendu chaque score interprétable
   et m'a évité de conclure à tort.
2. **Avancer une idée à la fois.** Un seul changement par modèle permet d'attribuer chaque gain ou chaque échec à sa
   cause.
3. **Confirmer avant d'annoncer.** Sans les trois graines, j'aurais présenté le meilleur essai d'Optuna (FID 68,0 à
   300 époques) comme un résultat acquis, sans savoir s'il se reproduisait.
4. **Un GAN se règle plus qu'il ne se conçoit.** À architecture constante, les hyperparamètres ont divisé le FID par
   plus de deux.

---

## 6. Pistes d'amélioration

| Piste | Ce qu'elle apporterait |
|---|---|
| Entraîner plus longtemps | les courbes de FID descendaient encore à 600 époques |
| Répéter les modèles 128 pixels et paysages sur plusieurs graines | savoir quelle part de leurs scores tient au hasard |
| Une recherche Optuna dédiée au 128 pixels | le réglage actuel a été trouvé en 64 pixels |
| Une architecture plus récente (StyleGAN2-ADA, attention) | des visages mieux construits, une meilleure diversité |
| Un GAN conditionnel sur le genre ou l'école | un seul modèle au lieu d'un par genre, et davantage de données par entraînement |
| D'autres genres, dont l'art abstrait proposé par le sujet | tester si la démarche tient hors de la peinture figurative |
| Une évaluation humaine (vrai ou généré ?) | une mesure directe du réalisme, indépendante d'Inception |
