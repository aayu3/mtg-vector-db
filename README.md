# PGVector Database for hosting MTGJSON data for use with LLMs

## Tables
Glossary, Cards, Rules
We will embed card descriptions + name as well as rules text
















Logic Flow

First Step
* Take in user prompt
* Use a lighter weight model to extract the following 
  * Card Names
  * Card Descriptions
    * Reference Glossary to indentify key terms extract
  * Maybe identify rules
  