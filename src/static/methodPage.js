function onClickMethod(endpoint, variables){
	const data = new FormData(); // form data for all data user provides

	// loop to iterate through all parameters and save them in formData
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
			data.append('image' , input.files[0]); // saving image to formData
			// post request to current endpoint
			$.ajax({
				url: endpoint,
				type:"POST",
				data: data,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){ // upload error handling
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
					alert("Upload error")
				},
				success: function(data){
					if(!data['success']){ // wrong parameters error handling
						alert(data['err'])
						return
					}

					bytestring = data['status']
					image = bytestring.split('\'')[1]
					resultImageBox.attr('src' , 'data:image/jpeg;base64,'+image) // display image from response

					if(data['extra']) // display extra data if they appear in response
						document.getElementById('extraResult').innerHTML = data['extra'];
				}
			});
		}
}


// method to display image provided by user
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