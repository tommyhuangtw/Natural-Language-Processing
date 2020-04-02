$(document).ready(() => {
  $('#classifyButton').click(e => {
    let form = document.createElement('form')
    document.body.appendChild(form)
    form.method = 'post'
    form.action = "/classifier2/analysis"
    let input = document.createElement('input')
    input.name = 'sentence'
    input.value = $("#inputSentence")[0].value
    form.appendChild(input)

    form.submit()
  })
})
