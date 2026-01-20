# LogNER: Hierarchical Log Parsing via Nested NER and MDL Optimization

**LogNER** is a novel log parsing framework designed to address the increasing complexity of cloud application logs. By utilizing **Nested Named Entity Recognition (NER)** and **Minimum Description Length (MDL)**, it achieves high parsing accuracy even for logs with deeply nested and complex variable structures.

## 🚀 Research Goal
* **Improve Accuracy**: Enhance log parsing and anomaly detection performance using Named Entity Recognition.
* **Handle Complexity**: Recognize nested variables (e.g., JSON, lists, maps) that existing parsers often fail to process correctly.
* **Optimal Selection**: Choose the best log template by comparing candidate templates based on their hierarchical structure.

---

## 💡 Motivations

### Limitations of Existing Parsers
* Existing parsers typically find variables in token units, often resulting in overly specific templates.
* They fail to consider multiple candidate templates for complex logs, leading to an explosion in the number of generated templates.

### The Need for Nested NER
* Modern logs contain nested variables where one variable includes others.
* LogNER uses Nested NER to identify all ranges of these nested variables to build a hierarchical tree of potential templates.

---

## 🛠 Solution Design

### 1. Template Generation via Nested NER
* The Nested NER model returns the range of all nested variables.
* These ranges are used to construct a **Tree**, where parent nodes represent broader variables and child nodes represent nested ones.
* By choosing different depths in the tree, we can generate both general and specific candidate templates.

### 2. MDL (Minimum Description Length) Cost
To select the "best" template set, LogNER calculates the MDL cost:
* **SRC (Static Representation Cost)**: The number of bits required to represent the template itself ($n \cdot ceil(\log_2 m)$).
* **DRC (Dynamic Representation Cost)**: The number of bits required to represent the log variables given a specific template.
* **Total Cost = SRC + DRC**: The template set with the lowest total MDL cost is selected.

### 3. Optimization for Large Datasets
To handle high time complexity ($O(m^n)$), we apply:
* **Compression**: Grouping logs with identical candidate templates to reduce redundant cases.
* **Graph-based Grouping**: Constructing an undirected graph of templates and processing mutually reachable groups independently.
* **Hierarchy-based Selection**: Dividing large groups into hierarchical levels to compare MDL costs efficiently.

---

## 📊 Evaluation Results

### MDL Cost Efficiency
LogNER demonstrates superior efficiency in template representation compared to traditional methods like Drain.

| Parser | SRC | DRC | **MDL Cost** |
| :--- | :--- | :--- | :--- |
| **LogNER** | **425.94** | **525.32** | **951.26** |
| Drain | 908.34 | 589.86 | 1497.2 |
*(Results based on MultiLog dataset)*

### Anomaly Detection Performance
In evaluations using the **MultiLog** dataset (KDD 2024), LogNER outperformed existing parsers.

* **DeepLog Integration**: LogNER prevents DeepLog from misidentifying normal sequences as anomalies (new events), a common issue with traditional parsers.
* **Supervised Learning**: Achieved high F1-scores (e.g., 85.7) when paired with models like Decision Trees and Logistic Regression.

---

## 🏫 Affiliation
* **Kyungpook National University (KNU)**, Daegu, Republic of Korea
* **Authors**: Minyeop Song, Eunwoo Go, Byungchul Tak
