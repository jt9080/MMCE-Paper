# MISSION
You will be given a social media post via the text and image of the post, we well as a candidate claim extracted from the post. Your task is to assess whether the claim is fully faithful to and entailed by the combined content of the image and text. This means that assuming the social media post is true, the extracted claim must also be true.

Do NOT check if the claim is true in reality — only whether it is faithful to the content of the image and text of the post.

Ignore whether the correct factual content had been extracted, focus on whether the extracted sentence is faithful (i.e. no hallucinations).

# EVALUATION CRITERIA
Classify the claim into one of three categories:
- **entailed**: The claim is fully aligned with the post content without any contradictions, hallucinations or unsupported additions.
- **partially_entailed**: The claim is partially aligned with the post content but contains minor variations or additional context not stated or implied in the post.
- **not_entailed**: The claim contains significant misaligned inferences, exaggerations beyond what's stated, major contradictions, hallucinations, or completely misaligns with the post content.

# RESPONSE FORMAT
Return the response in the following JSON format:
```json
{
    "classification": "entailed",
    "reasoning": "Brief explanation of the classification based on how well the claim aligns with the post content"
}
```

# INPUT
