# Legal-Sentence-Role-Classification
In US Case Law it is crucial that no judge overrules the decision of another judge. 
For legal work this means in consequence that, thousands of legal experts are employed only to scan decisions to find similar ones regarding a current case and hence, automatization of this task would be beneficial.
If one wants to automize such task, one approach would be to break each decision down into more granular sets of Information like Facts, Reasoning, Citations etc. and then compare those against the current case.
In this work we tackle this issue of classifying the rhetorical role of a sentence in a decision.

As a basis we use [a dataset consisting of US Board of Veteran's Appeals Decisions]( https://github.com/LLTLab/VetClaims-JSON ) to build a system that is capable of:
    1. splitting the decisions into sentences
    2. classifying those sentences into the classes Citation, Finding of Fact, Reasoning,…  

