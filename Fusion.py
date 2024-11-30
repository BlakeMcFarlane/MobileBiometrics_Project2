def score_level_fusion(face_scores, voice_scores, alpha=0.5):
    """
    Perform score-level fusion by weighted sum of face and voice scores.
    
    Parameters:
        face_scores (list): Scores from face recognition.
        voice_scores (list): Scores from voice recognition.
        alpha (float): Weight for fusion. Default is 0.5 (equal weighting).
    
    Returns:
        list: Fused scores.
    """
    return [alpha * face + (1 - alpha) * voice for face, voice in zip(face_scores, voice_scores)]

# Example fusion of scores
face_scores = [0.9, 0.8, 0.4, 0.6]
voice_scores = [0.7, 0.9, 0.5, 0.3]
fused_scores = score_level_fusion(face_scores, voice_scores)
print(f"Fused Scores: {fused_scores}")
