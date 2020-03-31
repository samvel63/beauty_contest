import vk


if __name__ == '__main__': 
    token = 'd8dcbc8ad8dcbc8ad8dcbc8a81d8ac1129dd8dcd8dcbc8a865fce6267efd854821bc783'
    session = vk.Session(access_token=token)  # Authorization
    vk_api = vk.API(session)
