function openFile(event) {
	var reader = new FileReader();
	reader.readAsText((event.target).files[0]);
    reader.onload = function() {

		 $("#t1").val(reader.result);
    };
	return false;
};

// La fonction fade_out permet de faire disparaitre un élément à partir de son id.
function fade_out(id) {
	to_remove = document.getElementById(id)
	to_remove.parentNode.removeChild(to_remove);
	return false;
}


// La fonction fade_out_children fait disparaitre tous les enfants de la div id.
function fade_out_children(id){
	parent = document.getElementById(id);
	while(parent.firstChild){
		parent.removeChild(parent.firstChild)
	}
	return false
}

// La fonction custom_alert permet de lancer une alerte qui sera contenue dans la div id_wrapper,
//avec le message message
function custom_alert(message,id_wrapper) {
	var alert = document.createElement('div');
	alert.classList.add('alert');
	alert.classList.add('alert-primary');
	alert.id = 'custom-alert';
	alert.innerHTML = message;
	var wrapper = document.getElementById(id_wrapper);
	document.body.appendChild(alert, wrapper);
	return false;
};


//top_n prend en entrée un array et renvoie un array contenant ses n premiers elements
function top_n(array,n){
	if (n>array.length) {
		return "Error : n is bigger than array size"
	}
	var sorted = Array.from(array);
	sorted.sort(function(a, b){return b - a});
	return sorted.slice(0,n);
}


//heat_color retourne une couleur pour la heat map des attentions en fonction de la valeur de l'attention
function heat_color(value){
	var l = Math.max((0.95 - 4*value),0.54) * 100;
	return "hsl(20, 88.9%," + l + "%)"
}

//attention créé la visualisation de l'attention dans la balise ol ol_id, à partir de l'ensemble des token et des attentions associées.
function attention(ol_id,tokens,attentions){
	var ol = document.getElementById(ol_id);
	var top_attentions = top_n(attentions,3);
	for (var i = 0; i<tokens.length; i++){

		//Création des contenants
		var li = document.createElement('li');
		var row = document.createElement('div');
		row.classList.add('row');


		//Disposition des tokens (phrases ici)
		var token = document.createElement('div');
		token.innerHTML = tokens[i];
		token.classList.add('col-7');
		if (attentions[i]>0.04 || top_attentions.includes(attentions[i])) {
			token.style['color'] = '#f26722';
		}
		token.style['margin']= 'auto';

		//Ajout d'un point pour signaler le lien entre texte et valeur de l'attention
		var point = document.createElement('span');
		point.classList.add('dot');

		//Disposition des couleurs et valeur d'attention
		var heat = document.createElement('div');
		heat.classList.add('col-4');
		heat.classList.add('heat');
		heat.style['background-image'] = 'linear-gradient(to left, white, ' + heat_color(attentions[i]) + ')';
		heat.innerHTML = Math.round(attentions[i]*10000)/10000;


		//Ils sont tous enfants de la div row que l'on va insérer dans la liste
		row.appendChild(token);
		row.appendChild(point)
		row.appendChild(heat);

		li.appendChild(row);
		ol.appendChild(li);
	}
	return false
}
