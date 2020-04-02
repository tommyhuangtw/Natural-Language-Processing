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

    fetch('/classifier2/predict', package)
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
        chart.data.datasets[0].backgroundColor = json.result[0] == 'T' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)'
        chart.data.datasets[0].borderColor = json.result[0] == 'T' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)'
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
        backgroundColor: result[0] == 'T' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)',
        borderColor: result[0] == 'T' ? 'rgb(255, 0, 0)' : 'rgb(0, 0, 255)',
        pointStyle: 'triangle',
        radius: 10
      }, {
        label: 'toxic',
        data: positiveCorpusVectors,
        backgroundColor: 'rgb(255, 150, 150)'
      }, {
        label: 'non-toxic',
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

  $('#positiveExample1').click(e => {
    $('#inputSentence')[0].value = "Your blocks do not deter me I may be blocked but you are still an asshole And cunt"
  })

  $('#positiveExample2').click(e => {
    $('#inputSentence')[0].value = "Oh God, Yeah the writer's think we're stupid, well sorry writers we know everything lol"
  })

  $('#negativeExample').click(e => {
    $('#inputSentence')[0].value = "How are you? Long time no see!!! Did you see how many MAdonna articles I promoted to GA?  talk2me please"
  })

  topWords = topWords.map(keyValue => [keyValue[0], keyValue[1] + 1])
  bottomWords = bottomWords.map(keyValue => [keyValue[0], -keyValue[1] + 1])
  WordCloud($('#positiveWordCloud')[0], { list: topWords, weightFactor: 15 } );
  WordCloud($('#negativeWordCloud')[0], { list: bottomWords, weightFactor: 15 } );
})
