from gliner import GLiNER
text = """
Dear Dr. Melissa,
I hope you are doing well. My name is John, and I am writing to request a consultation with you regarding a recent health concern that requires specialist evaluation.
I would appreciate it if you could let me know your earliest available appointment date. If there are any medical reports, scans, or documents you would like me to bring along, please let me know in advance.
Thank you for your time and consideration. I look forward to hearing from you soon.
Warm regards,
John
john@email.com
"""
labels = ["email", "phone_number", "user_name"]
model = GLiNER.from_pretrained("nvidia/gliner-pii")
entities = model.predict_entities(text, labels, threshold=0.5)
print (entities)


