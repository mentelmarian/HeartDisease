var chiavi
var selezionato = false;
var dataSelection=[];

var margin = {top: 20, right: 20, bottom: 110, left: 50},
  margin2 = {top: 430, right: 20, bottom: 30, left: 50},
  width = window.innerWidth/2 - margin.left - margin.right,
  height = 500 - margin.top - margin.bottom,
  height2 = 500 - margin2.top - margin2.bottom;

var parseDate = d3.timeParse("%b %Y");

var x = d3.scaleLinear().range([0, width]),
  x2 = d3.scaleLinear().range([0, width]),
  y = d3.scaleLinear().range([height, 0]),
  y2 = d3.scaleLinear().range([height2, 0]);

var xAxis = d3.axisBottom(x),
  xAxis2 = d3.axisBottom(x2),
  yAxis = d3.axisLeft(y);

var brush = d3.brushX()
  .extent([[0, 0], [width, height2]])
  .on("brush", brushed);
    
var brushTot=d3.brush()
  .extent([[0,0],[width, height]])
  .on("end", selected);
    
var focus;

var dati

var color= d3.scaleOrdinal(d3.schemeCategory10);

var clicked_disease = false;
var clicked_healthy = false;



/////////////////////////////////////////// START SCATTER  ///////////////////////////////////////////

function drawScatter(data){
  var svg = d3.select("#scatterArea").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom);

    

  svg.append("defs").append("clipPath")
    .attr("id", "clip")
    .append("rect")
    .attr("width", width)
    .attr("height", height);

  focus = svg.append("g")
    .attr("class", "focus")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var context = svg.append("g")
    .attr("class", "context")
    .attr("transform", "translate(" + margin2.left + "," + margin2.top + ")");
    

  x.domain(d3.extent(data, function(d) { return +d[chiavi[0]]; }));
  y.domain(d3.extent(data, function(d) { return +d[chiavi[1]]; }));
  x2.domain(x.domain());
  y2.domain(y.domain());
    
    

  // append scatter plot to main chart area 
  var dots = focus.append("g");
    dots.attr("clip-path", "url(#clip)");
    dots.selectAll("dot")
      .data(data)
      .enter().append("circle")
      .attr('class', 'dot')
      .attr("id", function(d){return d.id;})
      .attr("r",5)
      .attr("fill","grey")
      .attr("opacity",".3")
      .attr("cx", function(d) { return x(+d[chiavi[0]]); })
      .attr("cy", function(d) { return y(+d[chiavi[1]]); })
      .style("fill", function(d) {return color(d[chiavi[13]]); });
          
          
  focus.append("g")
    .attr("class", "axis axis--x")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

  focus.append("g")
    .attr("class", "axis axis--y")
    .call(yAxis);
        
  focus.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left)
    .attr("x",0 - (height / 2))
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .text(chiavi[1]);  
        
  svg.append("text")             
    .attr("transform",
          "translate(" + ((width + margin.right + margin.left)/2) + " ," + 
                        (height + margin.top + margin.bottom) + ")")
    .style("text-anchor", "middle")
    .text(chiavi[0]);
        
  focus.append("g")
    .attr("class", "brushT")
    .call(brushTot);
        
  // append scatter plot to brush chart area      
  var dots = context.append("g");
    dots.attr("clip-path", "url(#clip)");
    dots.selectAll("dot")
      .data(data)
      .enter().append("circle")
      .attr('class', 'dotContext')
      .attr("r",3)
      .style("opacity", .5)
      .attr("cx", function(d) { return x2(d[chiavi[0]]); })
      .attr("cy", function(d) { return y2(d[chiavi[1]]); })
      .style("fill", function(d) {return color(d[chiavi[13]]); });
          
  context.append("g")
    .attr("class", "axis axis--x")
    .attr("transform", "translate(0," + height2 + ")")
    .call(xAxis2);

  context.append("g")
    .attr("class", "brush")
    .call(brush)
    .call(brush.move, x.range());

  
  var legend = svg.selectAll(".legend")
    .data(["Heart Disease", "Healthy"])//hard coding the labels as the datset may have or may not have but legend should be complete.
    .enter().append("g")
    .attr("class", "legend")
    .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

  // draw legend colored circles
  legend.append("circle")
      .attr("cx", margin.left + 25)
      .attr("cy",30)
      .attr("r", 6)
      .style("fill", function(d){ 
        if (d == "Healthy"){
          return "rgb(31, 119, 180)";
        };
        return "rgb(255, 127, 14)";
      })
      .on("click", function(e){         // function to highlight when clicking on the legend
        if (e == "Heart Disease"){
          clicked_disease ? clicked_disease = false : clicked_disease = true;
          clicked_disease ? colorFromLegend("highlight disease") : colorFromLegend("undo disease");
        }
        else{
          clicked_healthy ? clicked_healthy = false : clicked_healthy = true;
          clicked_healthy ? colorFromLegend("highlight healthy") : colorFromLegend("undo healthy");
        }
      });
      

  // draw legend text
  legend.append("text")
      .attr("x", margin.left + 35)
      .attr("y", 30)
      .attr("dy", ".35em")
      .text(function(d) { return d;});
      
  
}

/////////////////////////////////////////// END SCATTER  ///////////////////////////////////////////

/////////////////////////////////////////// START PARALLEL  ///////////////////////////////////////////

function drawParallel(data){

  var margin = {top: 50, right: 40, bottom: 80, left: 40},
  width = window.innerWidth - margin.left - margin.right - 20,
  height = 500 - margin.top - margin.bottom;

  var x = d3.scaleBand().rangeRound([0, width+100]).padding(.1),
  y = {},
  dragging = {};


  var line = d3.line(),
  //axis = d3.axisLeft(x),
  background,
  foreground,
  extents;

  var svg = d3.select("#parallelArea").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


  // Extract the list of dimensions and create a scale for each.
  //pca.csv[0] contains the header elements, then for all elements in the header
  //different than "name" it creates and y axis in a dictionary by variable name
  x.domain(dimensions = d3.keys(data[0]).filter(function(d) {
      if ((d == "X") || (d == "Y")) {
          return false;
      }
      return y[d] = d3.scaleLinear()
          .domain(d3.extent(data, function(p) { 
              return +p[d]; }))
          .range([height, 0]);
    }));

  extents = dimensions.map(function(p) { return [0,0]; });

  // Add grey background lines for context.
  background = svg.append("g")
    .attr("class", "background")
    .selectAll("path")
    .data(data)
    .enter().append("path")
    .attr("class","backpath")
    .attr("d", path);

  // Add red foreground lines for focus.
  foreground = svg.append("g")
    .attr("class", "foreground")
    .selectAll("path")
    .data(data)
    .enter().append("path")
    .attr("class","forepath")
    .attr("d", path);

  // Add a group element for each dimension.
  var g = svg.selectAll(".dimension")
    .data(dimensions)
    .enter().append("g")
    .attr("class", "dimension")
    .attr("transform", function(d) {  return "translate(" + x(d) + ")"; })
    .call(d3.drag()
    .subject(function(d) { return {x: x(d)}; })
    .on("start", function(d) {
      dragging[d] = x(d);
      background.attr("visibility", "hidden");
    })
    .on("drag", function(d) {
      dragging[d] = Math.min(width, Math.max(0, d3.event.x));
      foreground.attr("d", path);
      dimensions.sort(function(a, b) { return position(a) - position(b); });
      x.domain(dimensions);
      g.attr("transform", function(d) { return "translate(" + position(d) + ")"; })
    })
    .on("end", function(d) {
      delete dragging[d];
      transition(d3.select(this)).attr("transform", "translate(" + x(d) + ")");
      transition(foreground).attr("d", path);
      background
        .attr("d", path)
        .transition()
        .delay(500)
        .duration(0)
        .attr("visibility", null);
    }));

  // Add an axis and title.
  g.append("g")
    .attr("class", "axis")
    .each(function(d) {  d3.select(this).call(d3.axisLeft(y[d]));})
    //text does not show up because previous line breaks somehow
    .append("text")
    .style("text-anchor", "middle")
    .attr("y", -9)
    .text(function(d) { return d; });

  // Add and store a brush for each axis.
  g.append("g")
    .attr("class", "brush")
    .each(function(d) {
      d3.select(this).call(y[d].brush = d3.brushY().extent([[-8, 0], [8,height]]).on("brush start", brushstart)
        .on("brush", go_brush).on("brush", brush_parallel_chart).on("end", brush_end));
    })
    .selectAll("rect")
    .attr("x", -8)
    .attr("width", 16);

  function resizeExtent(selection){
    selection
      .attr("x", -8)
      .attr("width", 16);
  }

  // Handles a brush event, toggling the display of foreground lines.
  function brush() {
    var actives = dimensions.filter(function(p) { return !y[p].brush.empty(); }),
    extents = actives.map(function(p) { return y[p].brush.extent(); });
    foreground.style("display", function(d) {
      return actives.every(function(p, i) {
        return extents[i].some(function(e){
          return e[0] <= d[p] && d[p] <= e[1];
        });
      }) ? null : "none";
    });
  }

  function position(d) {
    var v = dragging[d];
    return v == null ? x(d) : v;
  }

  function transition(g) {
    return g.transition().duration(500);
  }

  // Returns the path for a given data point.
  function path(d) {
    return line(dimensions.map(function(p) { return [position(p), y[p](d[p])]; }));
  }

  function go_brush() {
    d3.event.sourceEvent.stopPropagation();
  }

  function brushstart(selectionName) {
    foreground.style("display", "none")
    //console.log(selectionName);
    
    var dimensionsIndex = dimensions.indexOf(selectionName);

    //console.log(dimensionsIndex);
    
    extents[dimensionsIndex] = [0, 0];

    foreground.style("display", function(d) {
      
      return dimensions.every(function(p, i) {
          if(extents[i][0]==0 && extents[i][0]==0) {
              return true;
          }
        return extents[i][1] <= d[p] && d[p] <= extents[i][0];
      }) ? null : "none";
    });
  }

  // Handles a brush event, toggling the display of foreground lines.
  function brush_parallel_chart() {             /*selezione su parallel*/
    for(var i=0;i<dimensions.length;++i){
      if(d3.event.target==y[dimensions[i]].brush) {
        extents[i]=d3.event.selection.map(y[dimensions[i]].invert,y[dimensions[i]]);

      }
    }
        
    foreground.style("display", function(d) {
      return dimensions.every(function(p, i) {
        if(extents[i][0]==0 && extents[i][0]==0) {
          return true;
        }
          return extents[i][1] <= d[p] && d[p] <= extents[i][0];
      }) ? null : "none";
    });
  }

  function brush_end(){
    if (!d3.event.sourceEvent) return; // Only transition after input.
    
    
    if (!d3.event.selection){             // what to do on deselection (empty selection)
      reset_dots();
      console.log("reset sotto");
      resetTable();
      selezionato = false;
      return;
    }        
    console.log("selezione sotto");
      
    selezionato = true;
    reset_dots();                          // what to do after a selection
    resetTable();
    identify_and_color_dots(); 
    populateTable();   
    for(var i=0;i<dimensions.length;++i){
      if(d3.event.target==y[dimensions[i]].brush) {
        extents[i]=d3.event.selection.map(y[dimensions[i]].invert,y[dimensions[i]]);
        extents[i][0] = Math.round( extents[i][0] * 10 ) / 10;
        extents[i][1] = Math.round( extents[i][1] * 10 ) / 10;
        d3.select(this).transition().call(d3.event.target.move, extents[i].map(y[dimensions[i]]));
      }
    }
  }
  
}

/////////////////////////////////////////// END PARALLEL  ///////////////////////////////////////////

function associate_parallel_scatter() {
  d3.select("#parallelArea").selectAll(".forepath").attr("id",function(d){return d.id;});
}


d3.csv("csv/pca.csv", function(error, data) {

  chiavi= d3.keys(data[0])
  if (error) throw error;
  var l=data.length;
  for (i=0;i<l;i++)
    {
        data[i].id=i;
    }

  drawParallel(data)
  drawScatter(data)
  associate_parallel_scatter();

})

//create brush function redraw scatterplot with selection
function brushed() {
  var selection = d3.event.selection;
  console.log(selection)
  x.domain(selection.map(x2.invert, x2));
  focus.selectAll(".dot")
    .attr("cx", function(d) { return x(d[chiavi[0]]); })
    .attr("cy", function(d) { return y(d[chiavi[1]]); });
  focus.select(".axis--x").call(xAxis);
}

//modify the items selected on the scatter
function selected(){      
  dataSelection=[]
  var selection= d3.event.selection;
  
  //modify scatter according to selection on it
  
  
  if (selection != null && selezionato){        // unione delle selection  
    focus.selectAll(".dot")
      .style("opacity",function(d){
      if ((x(d[chiavi[0]]) > selection[0][0]) && (x(d[chiavi[0]]) < selection[1][0]) && (y(d[chiavi[1]]) > selection[0][1]) && (y(d[chiavi[1]]) < selection[1][1]) && selected_in_parallel(d.id)) {
        dataSelection.push(d.id)
        return "1"
      }
      else{
        return "0.3"
      }
      
    })
    resetTable();
    populateTable();        
  }
  else if (selection != null && !selezionato){    //selezione solo sopra
    focus.selectAll(".dot")
      .style("opacity",function(d){
      if ((x(d[chiavi[0]]) > selection[0][0]) && (x(d[chiavi[0]]) < selection[1][0]) && (y(d[chiavi[1]]) > selection[0][1]) && (y(d[chiavi[1]]) < selection[1][1])) {
        dataSelection.push(d.id)
        return "1"
      }
      else
      {
        return "0.3"
      }
      
      
    })
    console.log("selezione");
    resetTable();
    populateTable();
  }
  else
  {
    console.log("reset");
    resetTable();
    identify_and_color_dots();
    populateTable();
  }
  
  //modify parallel according to selection on scatter
  d3.select("#parallelArea").selectAll(".forepath")
    .style("stroke","#d41f37")

  if (selection != null){
    var c=d3.select("#parallelArea").selectAll(".forepath")
      .style("stroke",function(d){
        if ((x(d[chiavi[0]]) > selection[0][0]) && (x(d[chiavi[0]]) < selection[1][0]) && (y(d[chiavi[1]]) > selection[0][1]) && (y(d[chiavi[1]]) < selection[1][1])) {
          dataSelection.push(d.id)
          return "#d41f37"
        }
        else
        {
          return "none"
        }
      })
  //console.log(c)
  }
  
}

function identify_and_color_dots() {
  reset_dots();
  var selected_lines = [];
  var lines = document.getElementsByClassName('forepath');
  var selected_dots = [];
  var all_dots = document.getElementsByClassName('dot');
  
  for (var i = 0, n = lines.length; i < n; i++){
    if ((lines[i].getAttribute("style") == "" || lines[i].getAttribute("style") == "stroke: rgb(212, 31, 55);") && selezionato)
    {
        selected_lines.push(lines[i])
    }
  }
  for (var i = 0, n = all_dots.length; i < n; i++){
    for (var j = 0, m = selected_lines.length; j < m; j++){
      if (all_dots[i].getAttribute("id") == selected_lines[j].getAttribute("id")){
        all_dots[i].style.opacity = "1";
      }
    }
  }
}

function reset_dots(){
  var all_dots = document.getElementsByClassName('dot');
  for (var i = 0, n = all_dots.length; i < n; i++){
    all_dots[i].style.opacity = ".3";
  }
}

function selected_in_parallel(id) {
  var lines = document.getElementsByClassName('forepath');
  for (var i = 0, n = lines.length; i < n; i++){
    if (lines[i].getAttribute("id") == id && (lines[i].getAttribute("style") == "stroke: none;" || lines[i].getAttribute("style") == "" || lines[i].getAttribute("style") == "stroke: rgb(212, 31, 55);") && selezionato){
      return true;
    }
  }
  return false;
}

function populateTable(){
  var tbodyRef = document.getElementById('table').getElementsByTagName('tbody')[0];

  function addCell(tr, text) {
    var td = tr.insertCell();
    td.textContent = text;
    return td;
  }

  var all_dots = document.getElementsByClassName('dot');
  var highlighted = [];

  for (var i = 0, n = all_dots.length; i < n; i++){
    if (all_dots[i].getAttribute("style") == "fill: rgb(31, 119, 180); opacity: 1;" || all_dots[i].getAttribute("style") == "fill: rgb(255, 127, 14); opacity: 1;"){
      highlighted.push(all_dots[i]);
    }
  }
  
  let data = d3.selectAll(".dot").data();
  console.log(data[0]);
  
  // insert data
  data.forEach(function (item) {
    for (var j = 0, m = highlighted.length; j < m; j++){
      if (highlighted[j].getAttribute("id") == item.id){
        var row = table.insertRow();
        addCell(row, item.id);
        addCell(row, item["Age "]);
        addCell(row, item["Sex "] == 1 ? "Male" : "Female");
        var chest_pain_type;
        if(item["ChestPainType "] == 0){ chest_pain_type = "TA";}
        else if (item["ChestPainType "] == 1){chest_pain_type = "ATA";}
        else if (item["ChestPainType "] == 2){ chest_pain_type = "NAP";}
        else {chest_pain_type = "ASY";}
        addCell(row, chest_pain_type);
        addCell(row, item["RestingBP "]);
        addCell(row, item["Cholesterol "] == 0 ? "Not available" : item["Cholesterol "]);
        addCell(row, item["FastingBS "] == 1 ? "> 120 mg/dl" : "< 120 mg/dl");
        var resting;
        if(item["RestingECG "] == 0){ resting = "Normal";}
        else if (item["RestingECG "] == 1){resting = "ST";}
        else { resting = "LVH";}
        addCell(row, resting);
        addCell(row, item["MaxHR	 "]);
        addCell(row, item["ExerciseAngina "] == 1 ? "Yes" : "No");
        addCell(row, item["Oldpeak "]);
        var slope;
        if(item["ST_Slope "] == 0){ slope = "Down";}
        else if (item["ST_Slope "] == 1){slope = "Flat";}
        else { slope = "Up";}
        addCell(row, slope);
        addCell(row, item.HeartDisease == 1 ? "Heart Disease" : "No");
      }
      
    }
  });

}

function resetTable() {
  var Table = document.getElementById("table");
  Table.innerHTML = "<thead><tr><th>Id</th>" +
      "<th>Age</th>"+
      "<th>Sex</th>"+
      "<th>Chest Pain Type</th>"+
      "<th>Resting BP</th>"+
      "<th>Cholesterol</th>"+
      "<th>Fasting BS</th>"+
      "<th>Resting ECG</th>"+
      "<th>Max HR</th>"+
      "<th>Exercise Angina</th>"+
      "<th>Oldpeak</th>"+
      "<th>ST Slope</th>"+
      "<th>Heart Disease</th></tr></thead>";
}

function colorFromLegend(option){
  var all_dots = document.getElementsByClassName('dot');
  var lines = document.getElementsByClassName('forepath');
  var selected = [];

  if (option == "highlight disease"){
    for (var i = 0, n = all_dots.length; i < n; i++){
      if (all_dots[i].getAttribute("style") == "fill: rgb(255, 127, 14);" || all_dots[i].getAttribute("style") == "fill: rgb(255, 127, 14); opacity: 0.3;"){
        all_dots[i].style.opacity = "1";
        selected.push(all_dots[i].id);
      }
    }
    if (clicked_healthy){
      for (var i = 0, n = lines.length; i < n; i++){
        lines[i].style.stroke = "#d41f37";
      }
      resetTable();
      populateTable(); 
    } else{
      for (var i = 0, n = lines.length; i < n; i++){
        lines[i].style.stroke = "none";
      }
      for (var i = 0, n = lines.length; i < n; i++){
        console.log(lines[i].id);
        for (var j = 0, m = selected.length; j < m; j++){
          if (lines[i].id == selected[j]){
            lines[i].style.stroke = "#d41f37";
          }
        }
      }
      resetTable();
      populateTable();
    }    
  } else if (option == "undo disease"){
      for (var i = 0, n = all_dots.length; i < n; i++){
        if (all_dots[i].getAttribute("style") == "fill: rgb(255, 127, 14); opacity: 1;"){
          all_dots[i].style.opacity = "0.3";
          selected.push(all_dots[i].id);
        }
      }
      if (!clicked_healthy){
        for (var i = 0, n = lines.length; i < n; i++){
          lines[i].style.stroke = "#d41f37";
        }
        resetTable();
      }
      else{
        for (var i = 0, n = lines.length; i < n; i++){
          console.log(lines[i].id);
          for (var j = 0, m = selected.length; j < m; j++){
            if (lines[i].id == selected[j]){
              lines[i].style.stroke = "none";
            }
          }
        }
        resetTable();
        populateTable();
      }        
  } else if (option == "highlight healthy"){
      for (var i = 0, n = all_dots.length; i < n; i++){
        if (all_dots[i].getAttribute("style") == "fill: rgb(31, 119, 180);" || all_dots[i].getAttribute("style") == "fill: rgb(31, 119, 180); opacity: 0.3;"){
          all_dots[i].style.opacity = "1";
          selected.push(all_dots[i].id);
        }
      }
      if (clicked_disease){
        for (var i = 0, n = lines.length; i < n; i++){
          lines[i].style.stroke = "#d41f37";
        } 
        resetTable();
        populateTable();
      } else{
        for (var i = 0, n = lines.length; i < n; i++){
          lines[i].style.stroke = "none";
        }
        for (var i = 0, n = lines.length; i < n; i++){
          console.log(lines[i].id);
          for (var j = 0, m = selected.length; j < m; j++){
            if (lines[i].id == selected[j]){
              lines[i].style.stroke = "#d41f37";
            }
          }
        }
        resetTable();
        populateTable();
      } 
    } else {
    for (var i = 0, n = all_dots.length; i < n; i++){
      if (all_dots[i].getAttribute("style") == "fill: rgb(31, 119, 180); opacity: 1;"){
        all_dots[i].style.opacity = "0.3";
        selected.push(all_dots[i].id);
      }
    }
    if (!clicked_disease){
        for (var i = 0, n = lines.length; i < n; i++){
          lines[i].style.stroke = "#d41f37";
        }
        resetTable();
      }
      else{
        for (var i = 0, n = lines.length; i < n; i++){
          console.log(lines[i].id);
          for (var j = 0, m = selected.length; j < m; j++){
            if (lines[i].id == selected[j]){
              lines[i].style.stroke = "none";
            }
          }
        }
        resetTable();
        populateTable();
      } 
  }
}