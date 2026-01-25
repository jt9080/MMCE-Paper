# MISSION
You will be given a candidate claim that was extracted from a social media post. Your task is to assess whether this claim is understandable in isolation, without access to the original post or any external context. A decontextualized claim should be fully interpretable and self-contained to an average reader, who has no knowledge of the post.

# CRITICAL INSTRUCTIONS:
- Focus only on clarity and completeness of meaning. Do not check whether the claim is factually true or faithful to the post — only whether the claim can stand alone and be understood independently.

# Scoring Guidelines:
- **fully_decontextualized**: Understandable in isolation. The claim is completely self-contained, unambiguous, and requires no edits to be understood on its own. (Example: The mayor of NYC announced a new recycling program on June 1, 2024.)
- **partially_decontextualized**: The claim is mostly clear and contains some context, but has gaps, vague references or unresolved pronouns. The claim could benefit from some edits. (Example: Vaccination rates rose after that. -> could be rewrited to -> Vaccination in the UK rates rose after the 2023 campaign.)
- **not_decontextualized**: Not understandable in isolation. The claim cannot be interpreted on its own; key entities, referents, or context are missing. Major rewriting is needed. (Example: He did something yesterday.)

# RESPONSE FORMAT
Return the response in the following JSON format:
```json
{
    "classification": "fully_decontextualized",
    "reasoning": "Brief explanation of the classification"
}
```

# INPUT