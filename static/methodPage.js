function onClickMethod(endpoint, var1, var2){
	const formData = new FormData();
	var1Val = null
	var2Val = null
	if(var1)
		var1Val = document.getElementById(var1).value
	if(var2)
		var2Val = document.getElementById(var2).value

	if(var1 != 'undefined' && var1Val != null && var1Val != "")
		formData.append(var1, var1Val)
	
	if(var2 != 'undefined' && var2Val != null && var2Val != "")
		formData.append(var2, var2Val)

	resultImageBox = $('#resimagebox')
		input = $('#imageinput')[0]
		if(input.files && input.files[0])
		{
			formData.append('image' , input.files[0]);
			console.log(formData.get(var1))
			$.ajax({
				url: endpoint,
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					bytestring = data['status']
					image = bytestring.split('\'')[1]
					resultImageBox.attr('src' , 'data:image/jpeg;base64,'+image)
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