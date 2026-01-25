# MISSION
You will get two claims below, a generated claim and a reference claim. Your task is to perform a comprehensive similarity assessment between the generated claim and the reference claim, in the context of fact checking. Provide a similarity score from 1 to 4.

# CRITICAL INSTRUCTIONS:
- Be CONSISTENT in your scoring. Similar claims should always receive similar scores.
- Focus on the core implied meaning/content of the sentence, wording differences should be acceptable.
- Extra details that don't contradict the core claim is beneficial. Additional specific details (e.g. names, dates, locations, numbers) on the core factual assertion is a good thing. Do not penalize for verbosity and specificity.

# Scoring Guidelines:
- 1: Completely different, no overlap in themes, topics, or entities mentioned.
- 2: Minimal similarity in themes, topics, or entities mentioned, sentences have different meanings. The core factual statement to be fact checked is different.
- 3: Partial alignment in message conveyed, with significant differences that could potentially affect downstream fact checking.
- 4: Strong conceptual similarity with minor variations or near-identical meaning. 

# RESPONSE FORMAT
Return the response in the following JSON format:
```json
{
    "score": 3,
    "reasoning": "Brief explanation of why this score was given"
}
```

# INPUT
