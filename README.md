# What is "LogNER"?

- One line description of this project

    > "Log parser (Log template Extractor) for enhanced log analysis"

- Detailed description of the project

    This project introduces LogNER, a novel log parsing algorithm designed to address the logs with complex structure. Traditional parsers often fail to handle nested variables (e.g., JSON, lists, maps), leading to an explosion of fragmented and overly specific templates.

LogNER recognizing the hierarchical structure of log messages through Nested NER can lead to more accurate template extraction. To select the most accurate template set among various candidates, the project employs Minimum Description Length (MDL) optimization, balancing template simplicity and data representability.

The project consists of two main phases: 1) Hierarchical Log Parsing via Nested NER, and 2) Optimal Template Selection via MDL Cost Analysis.

## Index
[1.Architecture](#architecture)

[2.Getting Started](#getting-started)

[3.How to Run](#how-to-run)

  
## Architecture

- Research Design

    <img src="https://github.com/sominsong/NIMOS/blob/main/fig/research_archi.png">
    
    The figure above shows the methodology for extracting hierarchical log templates. It consists of two stages:

    1. Candidate Template Generation: NNER model returns the ranges of nested variables within a raw log. By using variable ranges, candidate templates are generated.

    2. Grouping: The goal is reducing number of cases by grouping relative logs (logs with same general template)
 
    3. Best Template Selection: The goal is to select the best template set among candidates to improve anomaly detection performance.
       - SRC(Scheme Representation Cost): Number of bits to represent the template itself.
       - DRC(Data Representation Cost): Number of bits to represent logs by the template set.
       - MDL Cost: Total costs (SRC + DRC).


## Getting started

```
sudo su
apt update && apt install -y make git-all

git clone 
```

## How to run

```

```

### Step 1. 

### Step 2.

### Step 3. 

### Step 4. 

### Step 5.

### Step 6. 

