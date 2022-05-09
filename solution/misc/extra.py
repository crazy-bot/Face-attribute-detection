def __calculate_norm_params(self, dataroot, X_train):
        psum, psum_sq = np.zeros((3)), np.zeros((3))
        pixel_count = 0
        for idx in X_train:
            frame_path = dataroot+'Tiny_Portrait_%06d.png'%idx
            frame = cv2.imread(frame_path)
            H, W, C = frame.shape
            pixel_count += (H + W)
            psum[0] += np.sum(frame[:,:,0])
            psum[1] += np.sum(frame[:,:,1])
            psum[2] += np.sum(frame[:,:,2])

            psum_sq[0] += np.sum(np.square(frame[:,:,0]))
            psum_sq[1] += np.sum(np.square(frame[:,:,1]))
            psum_sq[2] += np.sum(np.square(frame[:,:,2]))

        # mean and std
        total_mean = psum / pixel_count
        total_var  = (psum_sq / pixel_count) - (total_mean ** 2)
        total_std  = np.sqrt(total_var)

        return total_mean, total_std