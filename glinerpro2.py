from gliner import GLiNER
text = """
Subject: Payroll portal locked after MFA reset - URGENT
Ticket: INC-2025-10-24 * Opened: 2025-10-24 00:50 CST

Hi IAM team,

I can't access the myhome portal after resetting the MFA. I get the message that my account is locked. Please assist in unlocking my account so I can access the payroll information.
My username: tatabirlaambani, Email: tatabirlaambani@dreams11.com

Please help resole my issue as my work is getting impacted. I can be reached at 9876543210 for any further information.

Thanks,
Tata Birla Ambani
"""

labels = ["email", "phone_number", "user_name"]
model = GLiNER.from_pretrained("nvidia/gliner-pii")
entities = model.predict_entities(text, labels, threshold=0.8)
print (entities)