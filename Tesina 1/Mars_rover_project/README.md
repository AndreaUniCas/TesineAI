# Tesina 1 - Mars Rover Exploration Network

## Problema

La **Space Exploration Agency (SEA)** ha dispiegato una rete di **20 stazioni di esplorazione** sulla superficie di Marte. Un rover autonomo "**Pathfinder-X**" deve navigare tra queste stazioni per completare diverse missioni scientifiche, ottimizzando il percorso in base a vincoli specifici per ogni tipo di missione.

---

## Contesto Scientifico

Il rover opera in una regione marziana caratterizzata da:
- **Crateri** vulcanici (zone di difficile attraversamento)
- **Pianure** sabbiose (alte velocità ma rischio tempeste)
- **Canyon** rocciosi (protezione ma percorsi tortuosi)
- **Altipiani** (buona visibilità ma esposizione a radiazioni)

---

## Specifiche del Problema

- Rete composta da **20 nodi** (stazioni di esplorazione)
- Ogni nodo ha caratteristiche specifiche:
  - **Tipo di terreno** (crater, plain, canyon, plateau)
  - **Livello di radiazione** (0.0 - 1.0)
  - **Copertura solare** (0.0 - 1.0)
  - **Visibilità orbiter** (boolean)
- I costi di attraversamento variano in base a distanza e terreno

---

## Obiettivi

1. Implementare l'algoritmo **A\*** per la ricerca del percorso ottimale
2. Implementare l'algoritmo **BFS** per la ricerca del percorso più corto
3. Sviluppare **funzioni euristiche** specifiche per ogni scenario
4. Confrontare le performance dei due algoritmi

---

## Vincoli

- Il rover deve partire da un nodo START e raggiungere un nodo GOAL
- Considerare le caratteristiche specifiche di ogni scenario
- Rispettare i vincoli di sicurezza (radiazioni, energia, comunicazione)

---

## Fasi del Progetto

### Fase 1: Sviluppo degli Algoritmi

Implementare A* e BFS con euristiche personalizzate.

### Fase 2: Simulazione degli Scenari

---

#### 🔬 Scenario 1: Scientific Sample Collection

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Raccogliere campioni da siti geologici specifici |
| **Start Node** | 0 (Base Camp) |
| **Goal Node** | 13 (Mineral Site) |
| **Max radiation exposure** | 0.6 per nodo |
| **Vincoli** | Minimizzare esposizione totale a radiazioni |

**Funzione Euristica**:
```
h(n) = distance(n, goal) * (1 + radiation_level(n))
```

**Colori nodi**:
- 🔴 Rosso: Alta radiazione (> 0.6)
- 🟠 Arancione: Media radiazione (0.3 - 0.6)
- 🔵 Blu: Bassa radiazione (< 0.3)

---

#### ☀️ Scenario 2: Solar Power Route

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Percorso che massimizza ricarica batterie solari |
| **Start Node** | 16 (Water Cave - batteria scarica) |
| **Goal Node** | 5 (Observatory - stazione di ricarica) |
| **Min solar coverage** | 0.5 per nodo |
| **Vincoli** | Preferire zone ad alta esposizione solare |

**Funzione Euristica**:
```
h(n) = distance(n, goal) * (2.0 - solar_coverage(n))
```

**Colori nodi**:
- 🔴 Rosso: Bassa copertura solare (< 0.4)
- 🟠 Arancione: Media copertura (0.4 - 0.7)
- 🔵 Blu: Alta copertura solare (> 0.7)

---

#### 🛰️ Scenario 3: Communication Relay

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Mantenere comunicazione continua con orbiter |
| **Start Node** | 3 (Deep Crater - nessun segnale) |
| **Goal Node** | 11 (South Base - antenna principale) |
| **Required visibility** | Max 2 nodi consecutivi senza visibilità |
| **Vincoli** | Massimizzare nodi con visibilità orbiter |

**Funzione Euristica**:
```
h(n) = distance(n, goal) * (1.5 if not visible else 0.8)
```

**Colori nodi**:
- 🔵 Blu: Visibilità orbiter ✓
- 🔴 Rosso: Nessuna visibilità ✗

---

## Struttura della Rete

### Topologia Esagonale (Honeycomb)

La rete utilizza una topologia esagonale che rappresenta la disposizione naturale delle stazioni sulla superficie marziana.

### Caratteristiche dei Nodi

| Nodo | Tipo | Terreno | Radiazione | Solare | Visibilità |
|------|------|---------|------------|--------|------------|
| 0 | Base Camp | plateau | 0.20 | 0.90 | ✓ |
| 1 | Research Lab | plateau | 0.30 | 0.85 | ✓ |
| 2 | Crater Edge | crater | 0.70 | 0.40 | ✗ |
| 3 | Deep Crater | crater | 0.90 | 0.20 | ✗ |
| 4 | North Peak | plateau | 0.50 | 0.95 | ✓ |
| 5 | Observatory | plateau | 0.40 | 1.00 | ✓ |
| 6 | Canyon Entry | canyon | 0.30 | 0.50 | ✗ |
| 7 | East Ridge | plateau | 0.60 | 0.80 | ✓ |
| 8 | Dust Plains | plain | 0.40 | 0.70 | ✓ |
| 9 | Storm Zone | plain | 0.50 | 0.30 | ✗ |
| 10 | Ice Deposit | plain | 0.30 | 0.60 | ✓ |
| 11 | South Base | plateau | 0.20 | 0.85 | ✓ |
| 12 | Ancient River | canyon | 0.40 | 0.45 | ✗ |
| 13 | Mineral Site | canyon | 0.50 | 0.50 | ✗ |
| 14 | Volcanic Vent | crater | 0.80 | 0.30 | ✗ |
| 15 | Central Hub | plateau | 0.25 | 0.90 | ✓ |
| 16 | Water Cave | canyon | 0.35 | 0.10 | ✗ |
| 17 | West Outpost | plain | 0.45 | 0.75 | ✓ |
| 18 | Shelter Bay | canyon | 0.20 | 0.40 | ✗ |
| 19 | Crossroads | plain | 0.30 | 0.80 | ✓ |

---

## Output Richiesti per Ogni Scenario

- ✅ Percorso ottimale (sequenza di nodi)
- 📊 Costo totale del percorso
- 🔍 Numero di nodi espansi durante la ricerca
- 🔄 Confronto tra A* e BFS (efficienza, ottimalità)
- 📈 Visualizzazione grafica del percorso

---

## Criteri di Valutazione

| Criterio | Peso |
|----------|------|
| Correttezza implementazione **A*** | 25% |
| Correttezza implementazione **BFS** | 25% |
| Efficacia funzioni euristiche | 20% |
| Qualità del confronto tra algoritmi | 15% |
| Documentazione e report | 15% |

---

## File da Implementare

### Algoritmi di Pathfinding
- `src/algorithm/astar.py` - Implementare la classe `AStarPathfinder`
- `src/algorithm/bfs.py` - Implementare la classe `BFSPathfinder`

### Scenari
- `src/scenarios/scientific.py` - Implementare `solve_scientific_scenario()`
- `src/scenarios/solar.py` - Implementare `solve_solar_scenario()`
- `src/scenarios/communication.py` - Implementare `solve_communication_scenario()`

### Funzioni Euristiche (Opzionale)
- `src/algorithm/cost_functions.py` - Estendere le classi euristiche per ottimizzazioni avanzate

---

## Da Fornire prima dell'Orale

- 📁 **Codice completo** (repository GitHub)
- 📝 **Report** con:
  - Plot dei percorsi per ogni scenario
  - Tabella comparativa A* vs BFS
  - Analisi delle scelte euristiche
  - Discussione sui trade-off tra algoritmi

---

## Esecuzione

```bash
# Scenario Scientific Sample Collection
python main.py --scenario scientific --algorithm astar --start 0 --goal 13

# Scenario Solar Power Route
python main.py --scenario solar --algorithm astar --start 16 --goal 5

# Scenario Communication Relay
python main.py --scenario communication --algorithm bfs --start 3 --goal 11

# Confronto algoritmi (usare lo stesso scenario con algoritmi diversi)
python main.py --scenario scientific --algorithm astar --start 0 --goal 13
python main.py --scenario scientific --algorithm bfs --start 0 --goal 13

# Aiuto
python main.py --help
```

---

## Struttura del Progetto

```
Mars_Rover_Exploration/
├── README.md                     # Questo file
├── main.py                       # Entry point principale
├── requirements.txt              # Dipendenze Python
└── src/
    ├── algorithm/
    │   ├── astar.py              # TODO: Implementare A*
    │   ├── bfs.py                # TODO: Implementare BFS
    │   └── cost_functions.py     # Funzioni euristiche (fornite)
    ├── models/
    │   ├── network.py            # Classe MarsNetwork (fornita)
    │   └── node.py               # Classe MarsStation (fornita)
    ├── scenarios/
    │   ├── scientific.py         # TODO: Implementare scenario scientifico
    │   ├── solar.py              # TODO: Implementare scenario solare
    │   └── communication.py      # TODO: Implementare scenario comunicazione
    └── utils/
        └── visualization.py      # Utilità di visualizzazione (fornita)
```

---

## Dipendenze

Installare le dipendenze con:
```bash
pip install -r requirements.txt
```

Le dipendenze principali sono:
- `networkx` - Per la gestione dei grafi
- `matplotlib` - Per la visualizzazione
- `numpy` - Per i calcoli numerici