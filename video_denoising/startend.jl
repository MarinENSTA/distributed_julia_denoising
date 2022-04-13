function startend_calc(labindex, Nprocm, T)
	Q = T ÷ Nprocm
	R = T % Nprocm

	imagesNum = []
	i = 1
	while i <= Nprocm
		j = 0 

		if R != 0
			R -= 1
			j = 1
		end

		append!(imagesNum, Q+j)

		i+=1
	end


	reverse!(imagesNum)
	iend = 0
	istart = 1
	startend = []
	i = 1
	while i <= Nprocm
		iend = iend+imagesNum[i]
		append!(startend, (istart,iend))
		istart = iend+1
		i += 1
	end

	return  (startend[2 * labindex + 1], startend[2*(labindex+1)])
end