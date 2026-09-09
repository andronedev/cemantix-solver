# cemantix-solver

Solver pour [Cémantix](https://cemantix.certitudes.org) en Rust : un CLI qui joue seul contre l'API,
et une page web (WebAssembly, GitHub Pages) qui vous assiste pendant que vous jouez sur le site.

Cémantix renvoie pour chaque essai la similarité cosinus (arrondie à 4 décimales) entre l'essai et le mot secret,
calculée avec un word2vec fixe. Avec le même modèle en local, chaque score devient une contrainte exacte sur
le secret : on ne joue pas « chaud / froid », on **trilatère** dans l'espace des embeddings.

Résultat : **~2,9 coups en moyenne** (banc offline sur 300 parties, max 4), ~20 ms de calcul par partie.

## Modèle

Le serveur utilise `frWac_no_postag_phrase_500_cbow_cut10` de Jean-Philippe Fauconnier (frWac lemmatisé, CBOW,
500 dimensions, cut10, 2 Go). C'est vérifié : `cemantix verify` compare les 101 voisins publiés du mot d'hier
avec le calcul local et trouve 0 écart. Les variantes cut100 (200 ou 500 dims) ne correspondent pas.

Les phrases (`mot_mot`) sont ignorées au chargement (le serveur supprime les underscores) : il reste 440 157 mots.
Le cache `data/vecs.f32` fait 880 Mo ; le `.bin` de 2 Go peut être supprimé une fois les caches construits.

## Usage

```bash
cargo run --release -- fetch-model        # télécharge le modèle (2 Go, une fois) et construit les caches (~10 s)
cargo run --release -- verify             # vérifie que le modèle reproduit les scores du serveur
cargo run --release -- sim -n 300         # banc offline : distribution du nombre de coups
cargo run --release -- play               # résout le puzzle du jour
cargo run --release -- play --choose      # demande le mot de départ avant de lancer
cargo run --release -- play --start soleil --start chat   # force les premiers coups
cargo run --release -- play --dry-run urgent   # partie hors ligne contre un secret local
cargo run --release -- export-web         # exporte le modèle float16 pour la page web
```

Options de `play` : `--day N`, `--delay ms` (délai minimum entre deux appels API, 300 par défaut).
Avec `--choose` ou `--start`, le solver affiche pour votre mot le nombre de bits d'information attendus,
à comparer aux 12,5 bits de l'opener « mais ». Un mot absent du modèle est refusé et redemandé.

Exemple de partie :

```
#1  mais                 (opener précalculé)
    →    8.89°C 🥶   candidats 48965 → 26 (14 ms)
#2  métro                H=4.70 bits (26 tranches sur 26 candidats, 26 sondes, 0 ms)
    →   -0.04°C 🧊   candidats 26 → 1 (13 ms)
#3  pilier               H=0.00 bits (1 tranches sur 1 candidats, 1 sondes, 0 ms)
🥳 pilier trouvé en 3 coups (1.2 s)
```

## Page web (assistant)

`docs/` est une page statique servie par GitHub Pages : le cœur du solver compilé en WebAssembly plus les
48 965 mots plausibles en float16 (49 Mo, téléchargés une fois puis gardés dans le cache du navigateur).
La page propose un mot, vous le jouez sur Cémantix et reportez le °C affiché ; elle recalcule et propose
le suivant. Vous pouvez jouer un autre mot que la proposition (elle affiche les bits attendus des deux).

Pourquoi pas d'appel direct à l'API depuis la page ? Le serveur renvoie un faux `{"p":1000,"s":1}` à toute
requête dont l'en-tête `Origin` n'est pas le sien, et n'envoie pas d'en-têtes CORS. Le CLI forge cet en-tête,
un navigateur ne peut pas.

En local :

```bash
cargo run --release -- export-web                      # génère docs/model.f16, words.txt, meta.json
wasm-pack build web --target web --release --out-dir ../docs/pkg --no-typescript
(cd docs && python3 -m http.server 8765)               # puis http://127.0.0.1:8765
```

## Structure

- `core/` — bibliothèque pure (vecteurs, contraintes, entropie), sans I/O ; feature `parallel` = rayon.
- `cli/` — binaire `cemantix` : téléchargement du modèle, cache, API, banc, export web.
- `web/` — bindings wasm-bindgen autour de `core`.
- `docs/` — la page, le wasm et le modèle float16 (GitHub Pages).

## Algorithme

1. Matrice `V` (440 157 × 500, f32, lignes normalisées) ; masque `alive` des mots encore compatibles,
   initialisé aux ~49 000 mots « plausibles » (minuscules alphabétiques, rang de fréquence < 50 000).
2. Après un essai `g` de score `s` : `sims = V · V[g]` (un matvec parallèle, ~15 ms) puis
   `alive &= |sims − s| ≤ 1e-4`. Un essai découpe le vocabulaire en ~7 500 tranches.
3. Coup suivant = le candidat qui maximise l'entropie du partitionnement des candidats restants
   (style solver Wordle) ; premier coup = opener précalculé (`mais`, 12,5 bits).
4. Si plus aucun candidat : passage au vocabulaire complet, puis relâchement progressif de la tolérance.
   Un mot refusé par le serveur est retiré et rejoué sans compter.

Ce jeu est fait pour être joué : le solver n'envoie que quelques requêtes par partie, espacées.
