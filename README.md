# cemantix-solver

**Essayer dans le navigateur : [andronedev.github.io/cemantix-solver](https://andronedev.github.io/cemantix-solver/)**

Un solver pour [Cémantix](https://cemantix.certitudes.org) (et [QuelMot](https://quelmot.fr)), écrit en Rust.
Il trouve le mot du jour de Cémantix en trois coups, presque à chaque fois. QuelMot répond autre chose qu'un
cosinus, et demande un autre solveur : voir plus bas.

Il existe en deux formes :

- un **CLI** qui joue seul contre l'API de Cémantix ;
- une **page web** qui tourne entièrement dans le navigateur (le même code, compilé en WebAssembly) et vous
  souffle le prochain mot pendant que vous jouez sur le site : [andronedev.github.io/cemantix-solver](https://andronedev.github.io/cemantix-solver/).

```
#1  mais                 (opener précalculé)
    →    8.89°C 🥶   candidats 48965 → 26
#2  métro                H=4.70 bits (26 tranches sur 26 candidats)
    →   -0.04°C 🧊   candidats 26 → 1
#3  pilier               H=0.00 bits (dernier candidat)
🥳 pilier trouvé en 3 coups
```

## Pourquoi ça marche

Cémantix ne vous dit pas « chaud » ou « froid ». Il vous donne la similarité cosinus exacte, arrondie à quatre
décimales, entre votre mot et le mot secret, calculée dans un word2vec fixe. Si on a le même modèle sous la main,
chaque réponse devient une équation : le secret est un mot dont le cosinus avec « mais » vaut 0,0889 à 0,0001 près.
Sur 49 000 mots courants, une seule réponse ne laisse qu'une trentaine de candidats. Deux réponses, en général un seul.

On ne cherche donc pas à se rapprocher du mot, on **trilatère**. Le seul choix qui reste est celui du prochain mot à
jouer, et on prend celui qui coupe le mieux l'ensemble des candidats (le même critère d'entropie que les solvers de
Wordle). Le premier mot est précalculé : « mais » apporte 12,5 bits, presque le maximum possible.

Sur 300 parties simulées : 2,9 coups en moyenne, jamais plus de 4, une vingtaine de millisecondes de calcul par partie.

## Le modèle, une petite enquête

Le site indique seulement « données de Jean-Philippe Fauconnier », qui publie une douzaine de word2vec français.
Tout le monde cite le `frWac_non_lem_no_postag_no_phrase_200_cbow_cut100`, mais ses scores ne collent pas du tout.
Le bon est `frWac_no_postag_phrase_500_cbow_cut10` : lemmatisé, 500 dimensions, 2 Go, et il reproduit les scores
du serveur à la quatrième décimale. La commande `verify` refait ce test chaque jour à partir des voisins publiés du
mot de la veille.

Deux détails utiles sur l'API, découverts en chemin :

- toute requête dont l'en-tête `Origin` n'est pas celui du site reçoit un faux `{"p":1000,"s":1}`, quel que soit le
  mot. Le CLI met l'en-tête ; une page web ne peut pas, d'où la saisie manuelle dans l'assistant ;
- les rangs (‰) sont calculés sur un lexique filtré maison, pas sur tout le vocabulaire du modèle. Ils ne servent
  donc pas de contrainte.

## QuelMot ne donne pas un cosinus

Sa FAQ annonce un cosinus multiplié par 1000. C'est faux : le score est **un rang**.

    score = 1000 − rang du mot parmi les voisins du secret (dans leur lexique), plafonné à −999

Relevé le 9 septembre 2026, mot du jour « fantôme » :

| mot | score du site | 1000 − score | rang local (49 k mots) | rapport |
|---|---|---|---|---|
| télépathie (indice) | 399 | 601 | 402 | 1,50 |
| évocation (indice) | 266 | 734 | 482 | 1,52 |
| narrer (indice) | 133 | 867 | 570 | 1,52 |
| maison, esprit, parole | −999 | ≥ 1999 | 7 739 à 12 869 | plafond |

Les trois indices 🎁 du site sont des mots de rang 601, 734 et 867 : ils sont choisis par rang. Le rapport entre leur
rang et le nôtre est constant, environ 1,5, leur lexique faisant à peu près 74 000 mots contre nos 48 965. Et un mot
courant mais sans rapport tombe droit à −999.

Cela change tout. À Cémantix chaque réponse est une équation ; ici elle enferme le secret dans une *bande de rangs*,
et les trois quarts des mots ne renvoient que le plancher, c'est-à-dire rien. Comptez donc beaucoup plus de coups
qu'à Cémantix, et traitez les indices 🎁 comme ce qu'ils sont : des observations gratuites à rang connu, qui valent
chacune plusieurs coups. `sim --game quelmot` donne la distribution sur votre modèle.

Tester la contrainte « le mot g est au rang r des voisins de c » pour 49 000 candidats à chaque coup serait un
produit 49 k × 49 k par coup. On tabule donc une fois pour toutes, pour chaque mot, le cosinus de son 1er, 2e, 3e,
4e, 6e… 2000e voisin (18 niveaux, 36 octets par mot). Le rang devient alors une interpolation dans cette table, et
un coup coûte un produit matrice-vecteur, comme à Cémantix. Les niveaux sont resserrés exprès : avec des écarts de
×2,2 l'erreur d'interpolation atteignait ×1,4 au 99e centile et débordait la fenêtre de tolérance, contre ×1,08 avec
des écarts de ×1,5.

Le rapport de 1,5 entre les deux lexiques est le seul paramètre ajusté à la main. `check-ranks` le remesure à partir
d'une partie dont on connaît le mot :

```bash
cargo run --release -- check-ranks --game quelmot --secret fantôme \
    --obs télépathie=399 --obs évocation=266 --obs narrer=133 --obs maison=-999
```

Leur API (`scoreWord`, une fonction Firebase en europe-west1) réclame un jeton d'authentification anonyme, donc pas
de partie en direct depuis le CLI : la page web reste en saisie manuelle.

## Utilisation

Il faut Rust (édition 2024). La première commande télécharge le modèle, 2 Go, et construit un cache de 880 Mo en
une dizaine de secondes ; le `.bin` peut être supprimé ensuite.

```bash
cargo run --release -- fetch-model
cargo run --release -- verify               # le modèle local reproduit-il les scores du serveur ?
cargo run --release -- play                 # joue la partie du jour
cargo run --release -- play --choose        # vous choisissez le premier mot
cargo run --release -- play --start soleil  # ou vous l'imposez
cargo run --release -- sim -n 300           # banc offline, distribution du nombre de coups
cargo run --release -- hints pilier cactus  # l'échelle d'indices pour ces mots
cargo run --release -- hints --audit 300    # recalibrer les bandes d'indices
```

Pour les jeux à rangs, il faut d'abord la table des voisins. Elle demande une passe 49 k × 49 k, une poignée de
secondes avec rayon, et pèse 1,8 Mo :

```bash
cargo run --release -- build-ranks                       # table des rangs + opener dédié
cargo run --release -- play --game quelmot --dry-run pilier
cargo run --release -- sim --game quelmot -n 100         # banc du solveur par rangs
cargo run --release -- sim --game quelmot --jitter 0.15  # si les deux lexiques ne se dilatent pas également
cargo run --release -- sim --game quelmot --sim-alpha 1.8  # si le rapport de lexiques est mal estimé
```

Le CLI attend au moins 300 ms entre deux appels et n'en fait que trois ou quatre par partie. Ce jeu est fait pour
être joué ; l'intérêt ici est le problème, pas le classement.

## La page web

`docs/` est servi par GitHub Pages. Elle charge le solver en WebAssembly (156 Ko), les 48 965 mots plausibles en
float16 (49 Mo, une fois, puis gardés dans le cache du navigateur) et, pour QuelMot, la table des rangs (1,8 Mo).
Choisissez le jeu, jouez le mot proposé sur le site, reportez le score, recommencez. Vous pouvez jouer un autre mot
que la proposition : la page vous dit combien de bits il apporte par rapport à elle. En mode QuelMot, le bouton
« mot offert 🎁 » enregistre un mot donné par le site sans le compter comme un coup.

Si l'export ne contient pas `ranks.f16`, l'onglet QuelMot reste désactivé et Cémantix fonctionne normalement.

## Le mode Indices

L'assistant connaît la réponse bien avant vous, ce qui est exactement ce qu'on ne veut pas quand on joue. Le second
onglet, **Indices**, masque le mot proposé et la liste des candidats — la page n'affiche plus que leur nombre — et
ajoute un bouton qui livre un palier de plus à chaque clic : le champ sémantique de loin, le champ de plus près,
puis un mot très proche du secret qui n'est pas lui. Chaque palier révélé est figé, et un refus ne consomme rien.
Pendant la saisie, la page dit si votre mot colle mieux ou moins bien au champ des candidats que votre meilleur
essai — sans jamais dire s'il en est un, ce qui serait une réponse complète.

Deux mesures sur le vrai modèle fixent la construction. D'abord, le barycentre des candidats n'est pas un objet
sémantique : les candidats Cémantix forment la coquille des mots à un cosinus donné de l'opener, pas un amas, et
leur moyenne a une norme de 0,19 avec le secret au rang 30 000. Un mot d'indice est donc noté par le **minimum**
de ses cosinus à chaque candidat survivant, ce qui le rend vrai de tous par construction. Ensuite, aucun seuil de
cosinus absolu n'est calibrable : le premier voisin d'un mot se situe entre 0,45 et 0,69 selon le mot. Les bandes
sont donc des **positions** dans ce classement partagé :

| palier | positions | mots | cosinus médian au secret |
|---|---|---|---|
| champ, de loin | 200–900 | 4 | 0,288 |
| champ, de près | 25–120 | 3 | 0,392 |
| mot très proche | 2–12 | 1 | 0,513 |

Deux filtres sont indispensables. Le filtre de radical, sinon « blog » livre *blogueur* et « démonstration »
livre *démontrer* ; et un plafond de fréquence à 20 000, sinon les bandes remontent *nallet*, *net-iris*, *ziki*.
Enfin, aucun indice n'est donné tant que plus de huit candidats survivent, ni quand leur cosinus minimal deux à
deux tombe sous 0,30 : `{méfier, intellectuellement, chipoter}` n'ont pas de champ commun, et un indice sur « le »
champ serait un mensonge. En pratique Cémantix débloque l'échelle au deuxième score reporté ; QuelMot demande
quelques coups de plus. `cargo run -- hints --audit N` remesure tout cela après un changement de modèle.

Pour la reconstruire :

```bash
cargo run --release -- build-ranks     # sinon QuelMot restera masqué
cargo run --release -- export-web      # docs/model.f16, words.txt, ranks.f16, meta.json
wasm-pack build web --target web --release --out-dir ../docs/pkg --no-typescript
(cd docs && python3 -m http.server 8765)
```

## Organisation

```
core/   le solver, sans I/O : vecteurs, contraintes de cosinus et de rang, table des
        voisins, entropie (feature `parallel` = rayon)
cli/    le binaire : modèle, caches, API, bancs, export web
web/    les bindings wasm-bindgen autour de core
docs/   la page, le wasm, le modèle float16 et la table des rangs
```

## Merci

À Jean-Philippe Fauconnier pour les embeddings, à enigmatix pour Cémantix, et à David Turner pour Semantle,
dont tout cela découle.
