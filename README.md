# cemantix-solver

Un solver pour [Cémantix](https://cemantix.certitudes.org) (et [QuelMot](https://quelmot.fr)), écrit en Rust.
Il trouve le mot du jour en trois coups, presque à chaque fois.

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

QuelMot utilise le même modèle mais arrondit le cosinus à trois décimales (un entier entre -1000 et 1000). Le solver
prend cette précision en paramètre ; comptez un coup de plus.

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
cargo run --release -- sim --game quelmot   # même chose avec l'arrondi de QuelMot
```

Le CLI attend au moins 300 ms entre deux appels et n'en fait que trois ou quatre par partie. Ce jeu est fait pour
être joué ; l'intérêt ici est le problème, pas le classement.

## La page web

`docs/` est servi par GitHub Pages. Elle charge le solver en WebAssembly (117 Ko) et les 48 965 mots plausibles en
float16 (49 Mo, une fois, puis gardés dans le cache du navigateur). Choisissez le jeu, jouez le mot proposé sur le
site, reportez le score, recommencez. Vous pouvez jouer un autre mot que la proposition : la page vous dit combien
de bits il apporte par rapport à elle.

Pour la reconstruire :

```bash
cargo run --release -- export-web      # docs/model.f16, words.txt, meta.json
wasm-pack build web --target web --release --out-dir ../docs/pkg --no-typescript
(cd docs && python3 -m http.server 8765)
```

## Organisation

```
core/   le solver, sans I/O : vecteurs, contraintes, entropie (feature `parallel` = rayon)
cli/    le binaire : modèle, cache, API, banc, export web
web/    les bindings wasm-bindgen autour de core
docs/   la page, le wasm et le modèle float16
```

## Merci

À Jean-Philippe Fauconnier pour les embeddings, à enigmatix pour Cémantix, et à David Turner pour Semantle,
dont tout cela découle.
