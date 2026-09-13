# Sumarni izveštaj — AML detekcija na Elliptic datasetu

## Finalna tabela rezultata

| Model | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| Random Forest | 98.07% | 0.9751 | 0.7221 | 0.8297 | 20 | 301 |
| Extra Trees | 97.96% | 0.9745 | 0.7045 | 0.8178 | 20 | 320 |
| Bagging (RF) | 97.94% | 0.9501 | 0.7211 | 0.8199 | 41 | 302 |
| **Ensemble (RF+ET+Bag)** | **98.12%** | **0.9873** | 0.7193 | **0.8323** | **10** | 304 |
| GCN (2-layer) | 92.74% | 0.4584 | 0.6464 | 0.5364 | 827 | 383 |

> Sve metrike se odnose na **illicit klasu** (klasa od interesa).  
> Test set: time_step 35–49 (16,670 transakcija, od kojih 1,083 illicit).

---

## Opis koraka

### Učitavanje podataka
Elliptic dataset sadrži 203,769 Bitcoin transakcija opisanih sa 165 feature-a
(lokalni i agregacioni atributi čvora u grafu). Svaka transakcija ima vremensku
oznaku (time_step 1–49). Raspodela klasa: 2.2% illicit, 20.6% licit, 77.1% unknown.

### Priprema podataka
Transakcije sa oznakom `unknown` su isključene iz supervised learning pipeline-a.
Primenjen je **temporalni train/test split**: time_step 1–34 čine trening set
(29,894 primera), a time_step 35–49 čine test set (16,670 primera). Ovaj split
je identičan onom iz referentnog rada (Alarab et al. 2020) i obezbeđuje
uporedivost rezultata — nema "curenja" budućih podataka u trening.

### Baseline i dodatni modeli
Trenirani su: Random Forest, Extra Trees, Bagging (sa RF kao base estimator) i
AdaBoost. Svi modeli pokazuju visoku accuracy (~97-98%), ali je F1 na illicit
klasi umereniji (0.63–0.83) zbog prirodne neuravnoteženosti dataseta (~11.6%
illicit u trening setu). AdaBoost je najslabiji (F1=0.63, 580 FP).

### Ensemble model
Average probability ensemble kombinuje predikcije RF, Extra Trees i Bagging modela
usrednjavanjem verovatnoća, uz prag klasifikacije 0.5. Ensemble ostvaruje
**F1=0.8323** i **samo 10 false positiva** — precision od 98.7% znači da skoro
svaka transakcija označena kao illicit zaista jeste illicit. Ovo je ključno u
praksi, gde lažne uzbune nose visok operativni trošak.

### GCN (Graph Convolutional Network)
Kao četvrti pristup primenjen je dvo-slojni GCN (torch_geometric, GCNConv)
koji eksplicitno koristi grafovsku strukturu transakcija iz `edgelist` fajla.
Arhitektura: ulazni sloj (165 feature-a) → GCNConv(64, ReLU, dropout 0.3) →
GCNConv(1, sigmoid), trenirano 100 epoha sa Adam optimizerom i
`BCEWithLogitsLoss` sa ponderisanjem klase (pos_weight=7.63) zbog imbalansa.
Model konvergira (loss: 0.54 → 0.23, train F1 → 0.83), ali na test setu
postiže F1=0.54 — znatno slabije od ansambla stabala. Razlog je poznata
slabost plitkih GCN modela pri temporalnom shiftu: model uči agregirane
feature-e susednih čvorova iz trening perioda, ali ta agregacija ne
generalizuje dobro na transakcije iz kasnijih vremenskih koraka sa drugačijom
strukturom susedstva. Dublja arhitektura ili GraphSAGE sa inductive sampling
bili bi bolji pristup, ali izlaze van opsega ovog rada.

### Temporalna analiza
F1 ensembla po vremenskim koracima otkriva dramatičan pad na time_step 43–49
(F1 pada na ≈0). Ovo se poklapa sa poznatim "dark market shutdown" događajem
iz literature — promenom obrasca ilegalnih transakcija nakon gašenja dark
market platformi. Model treniran na ranijim obrascima nije u stanju da
generalizuje na novi tip ilicitnih aktivnosti.

---

## Grafici

- [ROC krive svih modela](roc_curves.png)
- [F1-score ensembla po vremenskom koraku](f1_per_timestep.png)

---

## Referenca

Alarab, I., Prakoonwit, S., & Nacer, M. I. (2020).
*Comparative Analysis Using Supervised Learning Methods for Anti-Money Laundering in Bitcoin.*
Proceedings of the 5th International Conference on Machine Learning Technologies (ICMLT 2020).
