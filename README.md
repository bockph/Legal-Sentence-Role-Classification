# Legal-Sentence-Role-Classification
In US Case Law it is important that no decision overrules another. Hence, thousands of legal experts are employed to only scan decisions to find similar ones regarding a current case. If one wants to automize such task it is necessary to break the problem down into smaller and compare different sets of Information like Facts, Reasoning, Citations etc.  To tackle this, therefore, we want to use a dataset consisting of US Board of Veteran's Appeals Decisions ( https://github.com/LLTLab/VetClaims-JSON ) to build a system that is capable of splitting the decisions into sentences + a classifier that predicts whether a sentence is a Citation, Finding of Fact, Reasoning,…    Depending on how we get along with this task, we would then in a third step use this classifier to implement some kind of active Learning approach, where the classifier would get also non annotated data to train - with the fallback that it can ask whether an uncertain prediction is correct or not and hence being able to train on a much bigger dataset.