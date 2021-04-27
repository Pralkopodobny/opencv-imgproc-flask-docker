function onClickMethod(endpoint, variables){
	const data = new FormData();

	for (let varbl of variables){
		if(varbl != ""){
			v = document.getElementById(varbl).value
			if(v != 'undefined' && v != null && v != "")
				data.append(varbl, v)
		}
	}

	resultImageBox = $('#resimagebox')
		input = $('#imageinput')[0]
		if(input.files && input.files[0])
		{
			data.append('image' , input.files[0]);
			$.ajax({
				url: endpoint,
				type:"POST",
				data: data,
				cache: false,
				processData:false,
				contentType:false,
				success: function(data){
					if(!data['success']){
						alert(data['err'])
						return
					}

					bytestring = data['status']
					image = bytestring.split('\'')[1]
					resultImageBox.attr('src' , 'data:image/jpeg;base64,'+image)

					if(data['extra'])
						document.getElementById('extraResult').innerHTML = data['extra'];
				}
			});
		}
}



function readUrl(input){
	imagebox = $('#imagebox')
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){
			imagebox.attr('src',e.target.result); 
		}
		reader.readAsDataURL(input.files[0]);
	}

	
}