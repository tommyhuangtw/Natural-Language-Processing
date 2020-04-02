$(document).ready(() => {
  $('#classifyButton').click(e => {
    const package = {
      method: 'POST',
      headers: new Headers({
        "Accept": "application/x-www-form-urlencoded;charset=utf-8",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
      }),
      body: 'sentence=' + $("#inputSentence")[0].value,
    }

    fetch('/classifier1/predict', package)
      .then(response => response.json())
      .then(json => {
        console.log(json)
        $('#result')[0].innerText = json.result + " with " + Math.round(json.confidence * 10) / 10 + "% confidence"
        $('#positiveOccurrences')[0].innerText = json.positiveOccurrences
        $('#negativeOccurrences')[0].innerText = json.negativeOccurrences
        $('#positiveCosineSimilarity')[0].innerText = json.positiveCosineSimilarity
        $('#negativeCosineSimilarity')[0].innerText = json.negativeCosineSimilarity
        chart.data.datasets[0].data[0] = {
          x: json.sentenceVector[0],
          y: json.sentenceVector[1],
        }
        chart.data.datasets[0].backgroundColor = json.result[0] == 'P' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)'
        chart.data.datasets[0].borderColor = json.result[0] == 'P' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)'
        chart.update()
      })
  })

  let ctx = $('#scatteredChart')[0].getContext('2d')
  let chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'current',
        data: [sentenceVector],
        backgroundColor: result[0] == 'P' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)',
        borderColor: result[0] == 'P' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)',
        pointStyle: 'triangle',
        radius: 10
      }, {
        label: 'positive',
        data: positiveCorpusVectors,
        backgroundColor: 'rgb(255, 150, 150)'
      }, {
        label: 'negative',
        data: negativeCorpusVectors,
        backgroundColor: 'rgb(150, 150, 255)'
      }]
    },
    options: {
      responsive: true,
      scales: {
        xAxes: [{
          // ticks: {
          //   suggestedMin: 0,
          // },
          type: 'linear',
          position: 'bottom',
          scaleLabel: {
            display: true,
            labelString: 'value after LDA'
          }
        }],
        yAxes: [{
          // ticks: {
          //   suggestedMin: 0,
          // },
          type: 'linear',
          scaleLabel: {
            display: true,
            labelString: 'confidence'
          }
        }]
      }
    }
  })

  $('#positiveExample').click(e => {
    $('#inputSentence')[0].value = "I would like to visit this place again. It servers amazing food and deserts. definitely will come next time come"
  })

  $('#negativeExample').click(e => {
    $('#inputSentence')[0].value = "SERVICE sucks here. Never will visit again. disappointing dining experience"
  })

  $('#neutralExample').click(e => {
    $('#inputSentence')[0].value = "It is good but not really satisfactory. There is still room for improvement"
  })

  $('#positiveOODExample').click(e => {
    $('#inputSentence')[0].value = "Indulgent, sure, but most of what it indulges in is tremendously great cinema, and some of the finest action sequences in the medium's history."
  })

  $('#negativeOODExample').click(e => {
    $('#inputSentence')[0].value = "Choreography is all that's on offer here. The increasingly convoluted fight scenes turn into lifeless spectacle in this ugly franchise."
  })

  $('#overConfidenceInPositiveExample').click(e => {
    $('#inputSentence')[0].value = "Summary:  Beautiful hotel, highly error-prone staff. It is rare to find a hotel as beautiful as this. The look is very modern which I happen to love."
  })

  $('#overConfidenceInNegativeExample1').click(e => {
    $('#inputSentence')[0].value = "Ok so this is not fast pizza so if you need it in ten minutes and don't really care what it tastes like then this may not be your place"
  })

  $('#overConfidenceInNegativeExample2').click(e => {
    $('#inputSentence')[0].value = "The service is just ok but the food is awesome"
  })

  topWords = topWords.map(keyValue => [keyValue[0], keyValue[1] + 1])
  bottomWords = bottomWords.map(keyValue => [keyValue[0], -keyValue[1] + 1])
  WordCloud($('#positiveWordCloud')[0], { list: topWords, weightFactor: 15 } );
  WordCloud($('#negativeWordCloud')[0], { list: bottomWords, weightFactor: 15 } );
})
