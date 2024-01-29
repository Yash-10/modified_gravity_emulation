for d in ['train', 'val', 'test']:
   ...:     for img in glob.glob(f'official_pix2pix_data_velDiv_F5n1_GR/{d}/*.npy'):
   ...:         #print(img)
   ...:         name = img.split('/')[-1].split('.')[0]
   ...:         #print(name)
   ...:         arr = np.load(img)
   ...:         gr = arr[:, :512]
   ...:         fr = arr[:, 512:]
   ...:         gr_1 = gr[:256, :256]
   ...:         gr_2 = gr[:256, 256:]
   ...:         gr_3 = gr[256:, :256]
   ...:         gr_4 = gr[256:, 256:]
   ...:         fr_1 = fr[:256, :256]
   ...:         fr_2 = fr[:256, 256:]
   ...:         fr_3 = fr[256:, :256]
   ...:         fr_4 = fr[256:, 256:]
   ...:         np.save(f'official_pix2pix_data_velDiv_F5n1_GR_256X256/{d}/{name}_1.npy', np.hstack((gr_1, fr_1)))
   ...:         np.save(f'official_pix2pix_data_velDiv_F5n1_GR_256X256/{d}/{name}_2.npy', np.hstack((gr_2, fr_2)))
   ...:         np.save(f'official_pix2pix_data_velDiv_F5n1_GR_256X256/{d}/{name}_3.npy', np.hstack((gr_3, fr_3)))
   ...:         np.save(f'official_pix2pix_data_velDiv_F5n1_GR_256X256/{d}/{name}_4.npy', np.hstack((gr_4, fr_4)))

