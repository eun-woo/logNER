# LogNER: Hierarchical Log Parsing via Nested NER and MDL Optimization

[cite_start]**LogNER** is a novel log parsing framework designed to address the increasing complexity of cloud application logs[cite: 7, 12]. [cite_start]By utilizing **Nested Named Entity Recognition (NER)** and **Minimum Description Length (MDL)**, it achieves high parsing accuracy even for logs with deeply nested and complex variable structures[cite: 12, 13, 101].

## 🚀 Research Goal
* [cite_start]**Improve Accuracy**: Enhance log parsing and anomaly detection performance using Named Entity Recognition[cite: 12].
* [cite_start]**Handle Complexity**: Recognize nested variables (e.g., JSON, lists, maps) that existing parsers often fail to process correctly[cite: 13, 30, 31].
* [cite_start]**Optimal Selection**: Choose the best log template by comparing candidate templates based on their hierarchical structure[cite: 65, 75].

---

## 💡 Motivations

### Limitations of Existing Parsers
* [cite_start]Existing parsers typically find variables in token units, often resulting in overly specific templates[cite: 17, 53, 54].
* [cite_start]They fail to consider multiple candidate templates for complex logs, leading to an explosion in the number of generated templates[cite: 55, 66].

### The Need for Nested NER
* [cite_start]Modern logs contain nested variables where one variable includes others[cite: 29, 62].
* [cite_start]LogNER uses Nested NER to identify all ranges of these nested variables to build a hierarchical tree of potential templates[cite: 69, 70].

---

## 🛠 Solution Design

### 1. Template Generation via Nested NER
* [cite_start]The Nested NER model returns the range of all nested variables[cite: 69].
* [cite_start]These ranges are used to construct a **Tree**, where parent nodes represent broader variables and child nodes represent nested ones[cite: 70, 71].
* [cite_start]By choosing different depths in the tree, we can generate both general and specific candidate templates[cite: 72, 75].

### 2. MDL (Minimum Description Length) Cost
[cite_start]To select the "best" template set, LogNER calculates the MDL cost[cite: 101]:
* [cite_start]**SRC (Static Representation Cost)**: The number of bits required to represent the template itself ($n \cdot ceil(\log_2 m)$)[cite: 133, 136].
* [cite_start]**DRC (Dynamic Representation Cost)**: The number of bits required to represent the log variables given a specific template[cite: 147, 151].
* [cite_start]**Total Cost = SRC + DRC**: The template set with the lowest total MDL cost is selected[cite: 181].

### 3. Optimization for Large Datasets
[cite_start]To handle high time complexity ($O(m^n)$), we apply[cite: 182, 185]:
* [cite_start]**Compression**: Grouping logs with identical candidate templates to reduce redundant cases[cite: 190, 206].
* [cite_start]**Graph-based Grouping**: Constructing an undirected graph of templates and processing mutually reachable groups independently[cite: 237, 240, 251].
* [cite_start]**Hierarchy-based Selection**: Dividing large groups into hierarchical levels to compare MDL costs efficiently[cite: 270, 275].

---

## 📊 Evaluation Results

### MDL Cost Efficiency
[cite_start]LogNER demonstrates superior efficiency in template representation compared to traditional methods like Drain[cite: 394].

| Parser | SRC | DRC | **MDL Cost** |
| :--- | :--- | :--- | :--- |
| **LogNER** | **425.94** | **525.32** | **951.26** |
| Drain | 908.34 | 589.86 | 1497.2 |
[cite_start]*(Results based on MultiLog dataset [cite: 379, 393, 394])*

### Anomaly Detection Performance
[cite_start]In evaluations using the **MultiLog** dataset (KDD 2024), LogNER outperformed existing parsers[cite: 302, 349, 374].

* [cite_start]**DeepLog Integration**: LogNER prevents DeepLog from misidentifying normal sequences as anomalies (new events), a common issue with traditional parsers[cite: 385, 386].
* [cite_start]**Supervised Learning**: Achieved high F1-scores (e.g., 85.7) when paired with models like Decision Trees and Logistic Regression[cite: 376, 377].

---

## 🏫 Affiliation
* [cite_start]**Kyungpook National University (KNU)**, Daegu, Republic of Korea [cite: 4]
* [cite_start]**Authors**: Minyeop Song, Eunwoo Go [cite: 3]
