from torchtext.data.functional import load_sp_model

sp_model = load_sp_model("m_user.model")
sp_model = load_sp_model(open("m_user.model", 'rb'))