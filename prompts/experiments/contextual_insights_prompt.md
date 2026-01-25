# MISSION
Analyze this social media post and provide contextual insights to help identify the main factual claim.

# CONTEXTUAL ANALYSIS
Focus on these key insights:
1. INTENT: What's the main purpose of the post? (inform/persuade/entertain/satire/etc.)
2. TONE: What's the emotional tone of the post? (serious/humorous/sarcastic/anger/etc.)
3. CONTEXT: What real-world events/issues does this relate to? Include specific details like dates, locations, people, organizations.
4. VISUAL_CONTEXT: What specific people, objects, locations, or events are shown in the image that provide context for the claim? 

# RESPONSE FORMAT
Return your analysis as a JSON object with the following structure:
```json
{
    "intent": "description of the poster's main purpose",
    "tone": "description of the emotional tone",
    "context": "brief context with specific details about real-world events/issues",
    "visual_context": "description of what's shown in the image that provides context"
}
```
