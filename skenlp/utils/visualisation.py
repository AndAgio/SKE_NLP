from typing import Iterable


try:
    from IPython.display import display, HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "word_attributions",
        "pred_prob",
        "pred_class",
        "true_class",
        "attr_score",
        "raw_input_ids",
    ]

    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        true_class,
        attr_score,
        raw_input_ids,
    ) -> None:
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.attr_score = attr_score
        self.raw_input_ids = raw_input_ids


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def text_visualization(datarecords: Iterable[VisualizationDataRecord],
                       plot: bool = False,
                       legend: bool = True) -> str:  # In quotes because this type doesn't exist in standalone mode
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(datarecord.true_class),
                    format_classname(
                        "{} ({})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname("{}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = "".join(dom)
    with open("try.html", "w") as file:
        file.write(html)
    if plot and HAS_IPYTHON:
        display(HTML(html))

    return html
